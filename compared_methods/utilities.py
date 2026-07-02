import random
import logging
import numpy as np
import torch
from types import MethodType
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from .evaluation import evaluate_perplexity

"""
Sets a fixed seed for the reproducibility of the results
"""
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

  logging.info(f"Seed for reproducibility: {seed}")


"""
Prints the number of parameters of the model
"""
def printModelStats(model, model_type):
  model_total_params = sum(p.numel() for p in model.parameters())
  main_model_total_params = sum(p.numel() for p in model.model.layers.parameters())
  logging.info(f"[{model_type}] Full number of parameters = {model_total_params}")
  logging.info(f"[{model_type}] Main model number of parameters = {main_model_total_params}")


def loadModel(model_name, cache_dir=None):
  logging.info("Loading the model")

  if (
    "llama" in model_name.lower()
    or "phi-3" in model_name.lower()
    or "qwen2" in model_name.lower()
  ):
    dtype = torch.bfloat16
  else:
    dtype = torch.float16

  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_cache=False,
    torch_dtype=dtype,
    device_map="cuda:0",
    cache_dir=cache_dir,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
  )

  return model

"""
Temporarily modifies the forward functions of the transformer's blocks or 
submodules (attention or MLP) to mask their computation based on the provided 
masks. Attentions or or FFNs marked as "1" in the masks are replaced with 
identity operations or zero outputs.

Args:
  model (torch.nn.Module): The transformer model to modify.
  attnMask (list[int]): A binary mask indicating which attention submodules 
                        to mask (1 = mask, 0 = keep).
  mlpMask (list[int]): A binary mask indicating which MLP submodules to mask 
                       (1 = mask, 0 = keep).
"""
def maskModel(model, attnMask, mlpMask):
  for i in range(len(attnMask)):
    if attnMask[i] == 1 and mlpMask[i] == 1:  # mask the entire block

      def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        return (hidden_states, None, None)

      model.model.layers[i].forward_bak = model.model.layers[i].forward
      model.model.layers[i].forward = MethodType(
        identity_forward, model.model.layers[i]
      )

    elif attnMask[i] == 1 and mlpMask[i] == 0:  # mask the attention only
      if model.config.model_type in ("phi", "phi3"):

        def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
          return torch.zeros_like(hidden_states), None
      else:

        def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
          return 0, None, None

      model.model.layers[i].self_attn.forward_bak = model.model.layers[
        i
      ].self_attn.forward
      model.model.layers[i].self_attn.forward = MethodType(
        identity_forward, model.model.layers[i].self_attn
      )

    elif attnMask[i] == 0 and mlpMask[i] == 1:  # mask the mlp only
      if model.config.model_type in ("phi", "phi3"):

        def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
          return torch.zeros_like(hidden_states)
      else:

        def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
          return 0

      model.model.layers[i].mlp.forward_bak = model.model.layers[i].mlp.forward
      model.model.layers[i].mlp.forward = MethodType(
        identity_forward, model.model.layers[i].mlp
      )

"""
Restores the original forward functions of the transformer's blocks or 
submodules (attention or MLP) after they have been masked.

Args:
  model (torch.nn.Module): The transformer model to restore.
  attnMask (list[int]): A binary mask indicating which attention submodules 
                        were previously masked (1 = masked, 0 = unmasked).
  mlpMask (list[int]): A binary mask indicating which MLP submodules were 
                       previously masked (1 = masked, 0 = unmasked).
"""
def unmaskModel(model, attnMask, mlpMask):
  for i in range(len(attnMask)):
    if attnMask[i] == 1 and mlpMask[i] == 1:  # unmask the entire block
      model.model.layers[i].forward = model.model.layers[i].forward_bak

    elif attnMask[i] == 1 and mlpMask[i] == 0:  # unmask attention only
      model.model.layers[i].self_attn.forward = model.model.layers[
        i
      ].self_attn.forward_bak

    elif attnMask[i] == 0 and mlpMask[i] == 1:  # unmask FFN only
      model.model.layers[i].mlp.forward = model.model.layers[i].mlp.forward_bak


"""
Splits the input dataset into a calibration dataset consisting of a specified 
number of samples, each with a fixed sequence length.

Args:
  dataset (Dataset): The input dataset containing text data.
  tokenizer (Tokenizer): Tokenizer used to convert text into input IDs.
  num_samples (int): Number of calibration samples to generate.
  seq_len (int): Length of each calibration sequence (default: 2048).
  seed (int): Random seed for shuffling the dataset (default: 0).

Returns:
  List[Tensor]: A list of tensors representing calibration samples, each 
                of shape (1, seq_len).
"""
def get_calibration(dataset, tokenizer, num_samples, seq_len=2048, seed=0):
  samples_indices = list(range(len(dataset)))
  if seed != 0:
    random.shuffle(samples_indices)

  samples = dataset.select(samples_indices)
  input_ids = tokenizer(
    "\n\n".join(samples["text"]), return_tensors="pt", add_special_tokens=False
  ).input_ids

  # Split the calibration into num_samples of seq_len length
  calibration = []
  for i in range(num_samples):
    calibration.append(input_ids[:, i * seq_len : (i + 1) * seq_len])

  return calibration


"""
Captures the intermediate outputs of a model's blocks when processing a 
calibration dataset. The outputs can focus on either the entire sequence or 
only the last token of each sequence.

Args:
  model (torch.nn.Module): The model whose intermediate outputs are to be 
                           computed.
  calibration_set (List[Tensor]): The calibration dataset, represented as 
                                  input ID tensors.
  last_token (bool): If True, captures outputs for the last token only; 
                     otherwise, captures outputs for the entire sequence.

Returns:
  List[Tensor]: A list of tensors representing the intermediate outputs of 
                the model's blocks for each calibration sample.
"""
def compute_intermediate_outputs(model, calibration_set, last_token):
  intermediate_outputs = []
  for ci in tqdm(range(len(calibration_set)), desc="Intermediate outputs"):
    model.intermediate_outputs_ci = []

    # Set a forward hook to extract the intermediate output
    def hook(module, input, output):
      if last_token:
        # focus on the similarity on the last token in each sequence
        model.intermediate_outputs_ci.append(output[0][0, -1].to("cpu"))
      else:
        # focus on the similarity on the entire sequence
        model.intermediate_outputs_ci.append(output[0][0].to("cpu"))

    hooks = []
    for layer in model.model.layers:
      hooks.append(
        layer.register_forward_hook(
          lambda module, input, output: hook(module, input, output)
        )
      )

    # Move input_ids to GPU + forward
    input_ids = calibration_set[ci].to(model.device)
    with torch.no_grad():
      _ = model(input_ids)

    intermediate_outputs.append(torch.stack(model.intermediate_outputs_ci))

    # Remove all hooks
    for hook in hooks:
      hook.remove()

  return intermediate_outputs

"""
Captures the hidden states at the MLP submodule of each transformer block for 
a given calibration sample. Hidden states are collected before the final 
linear transformation in the MLP (commonly the down projection).

Args:
  model (torch.nn.Module): The transformer model
  calibration_sample (torch.Tensor): A single calibration sample

Returns:
  Dict[int, Tensor]: A dictionary where keys are block indices, and values 
                     are the extracted hidden states for each layer.
"""
@torch.no_grad()
def get_mlp_hidden_state(model, calibration_sample):
  if model.config.model_type in ("llama", "mistral", "phi3", "qwen2"):
    for i, layer in enumerate(model.model.layers):
      layer.mlp.down_proj.original_index = i
  else:
    for i, layer in enumerate(model.model.layers):
      layer.mlp.fc2.original_index = i

  hidden_states = {}

  # Set a forward hook to extract the intermediate output
  def hook(module, input, output):
    hidden_states[module.original_index] = input[0][0].to("cpu")

  hooks = []
  for layer in model.model.layers:
    if model.config.model_type in ("llama", "mistral", "phi3", "qwen2"):
      last_linear = layer.mlp.down_proj
    else:
      last_linear = layer.mlp.fc2

    hooks.append(
      last_linear.register_forward_hook(
        lambda module, input, output: hook(module, input, output)
      )
    )

  # Move input_ids to GPU + forward
  input_ids = calibration_sample.to(model.device)
  with torch.no_grad():
    _ = model(input_ids)

  # Remove all hooks
  for hook in hooks:
    hook.remove()

  return hidden_states

"""
Captures both the inputs and outputs of the entire FFN submodules in a
transformer model for a given calibration sample.

Args:
  model (torch.nn.Module): The transformer model
  calibration_sample (torch.Tensor): A single calibration sample, represented 
                               as an input ID tensor

Returns:
  Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    - The first dictionary contains the FFN inputs for each block
    - The second dictionary contains the FFN outputs for each block
"""
@torch.no_grad()
def get_mlp_inputs_outputs(model, calibration_sample):
  for i, layer in enumerate(model.model.layers):
    layer.mlp.original_index = i

  mlp_inputs = {}
  mlp_outputs = {}

  # Set a forward hook to extract the intermediate output
  def hook(module, input, output):
    mlp_inputs[module.original_index] = input[0][0].to("cpu")
    mlp_outputs[module.original_index] = output[0].to("cpu")

  hooks = []
  for layer in model.model.layers:
    hooks.append(
      layer.mlp.register_forward_hook(
        lambda module, input, output: hook(module, input, output)
      )
    )

  # Move input_ids to GPU + forward
  input_ids = calibration_sample.to(model.device)
  with torch.no_grad():
    _ = model(input_ids)

  # Remove all hooks
  for hook in hooks:
    hook.remove()

  return mlp_inputs, mlp_outputs


@torch.no_grad()
def reset_mlps_shape(model):
  layers = model.model.layers
  for layer in layers:
    if model.config.model_type in ("llama", "mistral", "qwen2"):
      device = layer.mlp.gate_proj.weight.data.device
      dtype = layer.mlp.gate_proj.weight.data.dtype
      layer.mlp.gate_proj.weight.data = torch.zeros(
        (model.config.intermediate_size, model.config.hidden_size),
        device=device,
        dtype=dtype,
      )
      layer.mlp.up_proj.weight.data = torch.zeros(
        (model.config.intermediate_size, model.config.hidden_size),
        device=device,
        dtype=dtype,
      )
      layer.mlp.down_proj.weight.data = torch.zeros(
        (model.config.hidden_size, model.config.intermediate_size),
        device=device,
        dtype=dtype,
      )
    elif model.config.model_type == "phi3":
      # phi3 adopts a joint gate and up projection
      device = layer.mlp.gate_up_proj.weight.data.device
      dtype = layer.mlp.gate_up_proj.weight.data.dtype
      layer.mlp.gate_up_proj.weight.data = torch.zeros(
        (model.config.intermediate_size * 2, model.config.hidden_size),
        device=device,
        dtype=dtype,
      )
      layer.mlp.down_proj.weight.data = torch.zeros(
        (model.config.hidden_size, model.config.intermediate_size),
        device=device,
        dtype=dtype,
      )
    elif model.config.model_type == "phi":
      device = layer.mlp.fc1.weight.data.device
      dtype = layer.mlp.fc1.weight.data.dtype
      layer.mlp.fc1.weight.data = torch.zeros(
        (model.config.intermediate_size, model.config.hidden_size),
        device=device,
        dtype=dtype,
      )
      layer.mlp.fc1.bias.data = torch.zeros(
        (model.config.intermediate_size), device=device, dtype=dtype
      )
      layer.mlp.fc2.weight.data = torch.zeros(
        (model.config.hidden_size, model.config.intermediate_size),
        device=device,
        dtype=dtype,
      )
    else:
      logging.error(f"Error: {model.config.model_type} is not supported")
      exit(0)


"""
Prunes the FFN submodule of a specified block in the transformer model based on
a given binary mask. The mask determines which neurons in the hidden state of
the FFN are preserved.
This function is used for the first stage of 2SSP

Args:
  model (torch.nn.Module): The transformer model to be pruned
  mask (Tensor): A binary tensor of shape (intermediate_size), where 0 indicates 
                 neurons to preserve and 1 indicates neurons to prune.
  block_i (int): The index of the block to prune.
"""
@torch.no_grad()
def prune_mlp(model, mask, block_i):
    
  preserve_mask = torch.where(mask == 0)[0]
  new_intermediate_size = preserve_mask.size(0)
  layer = model.model.layers[block_i] 

  if model.config.model_type in ("llama", "mistral", "qwen2"):
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[preserve_mask]
    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[preserve_mask]
    layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:,preserve_mask]

    layer.mlp.gate_proj.weight.out_features = new_intermediate_size
    layer.mlp.gate_proj.out_features = new_intermediate_size
    layer.mlp.up_proj.weight.out_features = new_intermediate_size
    layer.mlp.up_proj.out_features = new_intermediate_size
    layer.mlp.down_proj.weight.in_features = new_intermediate_size
    layer.mlp.down_proj.in_features = new_intermediate_size
  elif model.config.model_type == "phi3":
    # phi3 adopts a joint gate and up projection
    gate_up_weights = layer.mlp.gate_up_proj.weight.data
    gate_weights, up_weights = gate_up_weights.chunk(2, dim=0)
    
    gate_weights = gate_weights[preserve_mask]
    up_weights = up_weights[preserve_mask]
    
    layer.mlp.gate_up_proj.weight.data = torch.cat([gate_weights, up_weights], dim=0)
    layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:,preserve_mask]
  elif model.config.model_type == "phi":
    layer.mlp.fc1.weight.data = layer.mlp.fc1.weight.data[preserve_mask]
    layer.mlp.fc1.bias.data = layer.mlp.fc1.bias.data[preserve_mask]
    layer.mlp.fc2.weight.data = layer.mlp.fc2.weight.data[:,preserve_mask]
  else:
    logging.error(f"Error: {model.config.model_type} is not supported")
    exit(0)


"""
Second-Stage Attention Pruning (https://arxiv.org/abs/2406.10594)

Performs pruning of attention submodules in transformer layers during the 
second stage of 2SSP. Iteratively removes the attention submodule with the
smallest impact on perplexity.

Args:
  model (torch.nn.Module): The transformer model to prune.
  num_prune (int): The number of attention submodules to prune.
  calibration_input_ids (torch.Tensor): A calibration input tensor used to 
                          calculate perplexity and evaluate pruning decisions.

Returns:
  tuple[list[int], list[int]]: Two binary masks representing the pruning 
                               decisions for attention and MLP submodules 
                               for each block. For MLP the mask is all 0.
"""
@torch.no_grad()
def second_stage_attention(model, num_prune, calibration_input_ids):
  num_blocks = len(model.model.layers)
  attnMask = [0] * num_blocks
  mlpMask = [0] * num_blocks
  
  ppl = evaluate_perplexity(model, calibration_input_ids, seq_len=512, enable_tqdm=False)
  logging.debug(f"Original perplexity: {ppl}")

  for _ in tqdm(range(num_prune), desc="Second stage"):

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
      ppl = evaluate_perplexity(model, calibration_input_ids, seq_len=512, enable_tqdm=False)

      logging.debug(f"[Attention] When pruning {to_prune} perplexity is {ppl}")
      if ppl < best_ppl:
        best_ppl = ppl
        best_to_prune = to_prune

      # Unmask the model
      unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)
      attnMask[to_prune] = 0

    logging.debug(f"[Attention] Best to prune: {best_to_prune} ({best_ppl})")
    logging.debug("========================")

    attnMask[best_to_prune] = 1
    
    if model.config.model_type in ("llama", "mistral", "qwen2"):
      del model.model.layers[best_to_prune].self_attn.q_proj
      del model.model.layers[best_to_prune].self_attn.k_proj
      del model.model.layers[best_to_prune].self_attn.v_proj
      del model.model.layers[best_to_prune].self_attn.o_proj
    elif model.config.model_type == "phi3":
      del model.model.layers[best_to_prune].self_attn.qkv_proj
      del model.model.layers[best_to_prune].self_attn.o_proj
    elif model.config.model_type == "phi":
      del model.model.layers[best_to_prune].self_attn.q_proj
      del model.model.layers[best_to_prune].self_attn.k_proj
      del model.model.layers[best_to_prune].self_attn.v_proj
      del model.model.layers[best_to_prune].self_attn.dense
    else:
      logging.error(f"Error: {model.config.model_type} is not supported")
      exit(0)

  return attnMask, mlpMask
