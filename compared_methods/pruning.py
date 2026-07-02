from tqdm import tqdm
import torch
import logging

from .utilities import (
  compute_intermediate_outputs,
  maskModel,
  unmaskModel,
  get_mlp_hidden_state,
  prune_mlp,
  second_stage_attention
)
from .evaluation import evaluate_perplexity


"""
Sliding Window Cosine Similarity (https://arxiv.org/abs/2403.17887)

Args:
  model (torch.nn.Module): The transformer model to prune
  num_prune (int): The number of blocks to prune
  calibration_dataset (Iterable): A dataset used to compute intermediate outputs 
                                    for similarity comparisons.

Returns:
  list[int]: A binary mask representing the pruning decision for each block. 
"""
@torch.no_grad()
def window_based(model, num_prune, calibration_dataset):
  num_blocks = len(model.model.layers)
  intermediate_outputs = compute_intermediate_outputs(model, calibration_dataset, last_token=True)

  l1 = 0
  l2 = 1
  best_sim = 0

  for layer1 in tqdm(range(num_blocks), desc="Layer similarity"):
    layer2 = layer1 + num_prune

    if layer2 >= num_blocks:
      continue

    sims = [
      torch.nn.functional.cosine_similarity(
        intermediate_outputs[ci][layer1], intermediate_outputs[ci][layer2], dim=0
      ).item()
      for ci in range(len(intermediate_outputs))
    ]

    # Average similarity among all the calibration samples
    sim = sum(sims) / len(sims)

    if sim >= best_sim:
      best_sim = sim
      l1 = layer1
      l2 = layer2
  
  # Remove from l1+1 to l2 extremes included
  to_prune = list(range(l1 + 1, l2 + 1))
  mask = [0] * num_blocks
  for i in to_prune:
    mask[i] = 1

  return mask



"""
ShortGPT (https://arxiv.org/abs/2403.03853)

Args:
  model (torch.nn.Module): The transformer model to prune
  num_prune (int): The number of blocks to prune.
  calibration_dataset (Iterable): A dataset used to compute intermediate outputs 
                                   for similarity comparisons.

Returns:
  list[int]: A binary mask representing the pruning decision for each block.
.
"""
@torch.no_grad()
def shortGPT(model, num_prune, calibration_dataset):

  intermediate_outputs = compute_intermediate_outputs(model, calibration_dataset, last_token=True)
  num_blocks = intermediate_outputs[0].size(0)

  block_similarity = [0.0] * num_blocks

  # average over all the calibration samples
  for ci in range(len(intermediate_outputs)):
    for li in range(1,num_blocks):
      # the influence of a block is based on the cosine similarity between the
      # input and the output of the block
      block_similarity[li] += torch.nn.functional.cosine_similarity(
        intermediate_outputs[ci][li-1], intermediate_outputs[ci][li], dim=0
      ).item()

  # average
  block_influence = [ 1 - bi / len(intermediate_outputs) for bi in block_similarity ]

  # prune the layers with the lowest influence 
  to_prune = sorted(range(len(block_influence)), key=lambda i: block_influence[i])[:num_prune]
  mask = [0] * num_blocks
  for i in to_prune:
    mask[i] = 1

  return mask


"""
BlockPruner (https://arxiv.org/abs/2406.10594)

Iteratively removes the submodule of a block (attention or MLP) that results in 
the smallest reduction in perplexity, calculated using only one calibration sample.

Args:
  model (torch.nn.Module): The transformer model to prune
  num_prune (int): The number of equivalent entire blocks to prune 
  calibration_sample (torch.Tensor): A single calibration sample used to 
                          calculate perplexity for evaluating pruning decisions.

Returns:
  tuple[list[int], list[int]]: Two binary masks representing the pruning decision 
                               for attention and MLP submodules for each layer.
"""
@torch.no_grad()
def blockpruner(model, num_prune, calibration_sample):
  num_blocks = len(model.model.layers)
  attnMask = [0] * num_blocks
  mlpMask = [0] * num_blocks
    
  for _ in range(num_prune):

    # Find the best attention to prune
    best_to_prune = None
    best_ppl = float("inf")

    for to_prune in range(num_blocks):

      # Cannot prune a block twice
      if attnMask[to_prune] == 1:
         continue
      
      # mask the model
      attnMask[to_prune] = 1
      maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

      # Evaluate
      ppl = evaluate_perplexity(model, calibration_sample, seq_len=512, enable_tqdm=False)

      logging.debug(f"[Attention] When pruning {to_prune} perplexity is {ppl}")
      if ppl < best_ppl:
        best_ppl = ppl
        best_to_prune = to_prune

      # Unmask the model
      unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)
      attnMask[to_prune] = 0
      

    logging.debug(f"[Attention] Best to prune: {best_to_prune} ({best_ppl})")
    attnMask[best_to_prune] = 1

    # Find the best MLP to prune
    best_to_prune = None
    best_ppl = float("inf")

    for to_prune in range(num_blocks):

      # Cannot prune a block twice
      if mlpMask[to_prune] == 1:
         continue
      
      # mask the model
      mlpMask[to_prune] = 1
      maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

      # Evaluate
      ppl = evaluate_perplexity(model, calibration_sample, seq_len=512, enable_tqdm=False)

      logging.debug(f"[MLP] When pruning {to_prune} perplexity is {ppl}")
      if ppl < best_ppl:
        best_ppl = ppl
        best_to_prune = to_prune

      # Unmask the model
      unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)
      mlpMask[to_prune] = 0
      

    logging.debug(f"[MLP] Best to prune: {best_to_prune} ({best_ppl})")
    mlpMask[best_to_prune] = 1

  return attnMask, mlpMask




"""
2SSP: A Two-Stage Framework for Structured Pruning of LLMs (our method)

Implements the complete framework. In the first stage, MLP submodules are pruned
based on the L2 norms of their hidden states. In the second stage, attention
submodules are pruned iteratively to minimize perplexity loss.

Args:
  model (torch.nn.Module): The transformer model to prune. calibration_dataset
  (Iterable[torch.Tensor]): A dataset of samples used for calculating norms and
                        perplexity during pruning.
  pruning_rate (float): The overall proportion of parameters to prune, 
                        relative to the total number of parameters in the
                        model's main layers.
  num_attn_submodules_to_prune (int, optional): The number of attention
                        submodules to prune. If None, this is calculated based
                        on Equation 5 in the paper
  alpha (int, optional): balancing between the number of parameters pruned in
                        the first and the second stages. See "Choice of alpha
                        parameter" in Section "4.3. Hyperparameter Tuning" of
                        the paper (Default: 1.5)
  num_calibration_second_stage (int, optional): the number of calibration
                        samples to use for the second stage (Default: 1)

Returns:
  torch.nn.Module: The pruned transformer model, with both MLP and attention 
                   submodules pruned.
"""
@torch.no_grad()
def two_stage_2ssp(model, calibration_dataset, pruning_rate, num_attn_submodules_to_prune=None, alpha=1.5, num_calibration_second_stage = 1):
    
    layers = model.model.layers
    num_blocks = len(layers)

    logging.debug("======================")
    model_total_params = sum(p.numel() for p in model.parameters())
    main_model_total_params = sum(p.numel() for p in model.model.layers.parameters())
    attn_total_params = sum(p.numel() for p in model.model.layers[0].self_attn.parameters())
    mlp_total_params = sum(p.numel() for p in model.model.layers[0].mlp.parameters())
    logging.debug(f"[Original model] Full number of parameters = {model_total_params}")
    logging.debug(f"[Original model] Main model number of parameters = {main_model_total_params}")
    logging.debug(f"Attention parameters (one block): {attn_total_params}")
    logging.debug(f"MLP parameters (one block): {mlp_total_params}")
    logging.debug("======================")

    if num_attn_submodules_to_prune is None:
      num_attn_submodules_to_prune = round(
        num_blocks * pow(pruning_rate, (mlp_total_params / attn_total_params) / alpha)
      )
    logging.info(f"Pruning {num_attn_submodules_to_prune} attention submodules")


    if (num_attn_submodules_to_prune * attn_total_params) / main_model_total_params > pruning_rate:
      logging.error("Exceeded pruning parameters number")
      return False
      
    if (num_attn_submodules_to_prune * attn_total_params + num_blocks * mlp_total_params) / main_model_total_params < pruning_rate:
      logging.error(f"Unable to reach the target sparsity rate with only {num_attn_submodules_to_prune} pruned attention submodules")
      return False


    # 1. First stage: pruning the FFN neurons
    parameters_pruned_for_attention = num_attn_submodules_to_prune * attn_total_params
    target_parameters_to_prune = int(round(pruning_rate * main_model_total_params))
    mlp_params_to_prune = int(round((target_parameters_to_prune - parameters_pruned_for_attention) / num_blocks))  # number of remaining parameters to prune in each single mlp block
    mlp_pruning_rate =  mlp_params_to_prune / mlp_total_params 

    mlp_hidden_size = model.config.intermediate_size
    num_preserve_mlp = int(round(mlp_hidden_size * (1 - mlp_pruning_rate)))


    average_norms = [0] * num_blocks
    for sample in tqdm(calibration_dataset, total=len(calibration_dataset), desc="First stage"):
      hidden_states_ci = get_mlp_hidden_state(model, sample)

      for li in range(num_blocks):
        norm_ci_li = hidden_states_ci[li].norm(dim=0, p=2)
        average_norms[li] += norm_ci_li
    
    for li in range(num_blocks):
      average_norms[li] /= len(calibration_dataset)
      _, top_indices = torch.topk(average_norms[li], num_preserve_mlp)
      mask = torch.ones_like(average_norms[li])
      mask[top_indices] = 0

      prune_mlp(model, mask, li)  

    model.config.intermediate_size = num_preserve_mlp

    # 2. Second stage: pruning the Attention submodules
    calibration_input_ids = torch.cat(calibration_dataset[:num_calibration_second_stage], dim=1)
    
    attnMask, mlpMask = second_stage_attention(model, num_prune=num_attn_submodules_to_prune, calibration_input_ids=calibration_input_ids)
    maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

    return model
