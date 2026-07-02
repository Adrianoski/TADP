# The following code was sourced and adapted from the repository:
# https://github.com/IST-DASLab/EvoPress
import random
import copy
import numpy as np
from typing import List, Optional
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from .utilities import *


def collect_samples_with_join(
  calibration_dataset,
  tokenizer: AutoTokenizer,
  num_samples: int,
  sequence_length: int,
  text_key: str = "text",
):
  data = []
  data_iter = iter(calibration_dataset)

  samples_collected = 0
  current_sample = torch.tensor([], dtype=torch.int64)
  for sample in data_iter:
    tokenized_sample = tokenizer(
      sample[text_key], return_tensors="pt", add_special_tokens=False
    ).input_ids
    current_sample = torch.cat([current_sample, tokenized_sample], dim=1)
    if current_sample.numel() >= sequence_length:
      samples_collected += 1
      data.append(
        current_sample[:, :sequence_length]
      )  # trim to sequence length; this introduces bias
      current_sample = torch.tensor([], dtype=torch.int64)  # reset current sample
    else:
      # add 2 new lines to the current sample
      current_sample = torch.cat(
        [
          current_sample,
          tokenizer("\n\n", return_tensors="pt", add_special_tokens=False).input_ids,
        ],
        dim=1,
      )
    # Stop if sufficient number of samples are collected
    if samples_collected >= num_samples:
      break
  return data


@torch.no_grad()
def compute_kl_div(model, data, target_logits, batch_size=1):
  num_samples = len(data)
  device = next(model.parameters()).device
  # Running estimate of negative log-likelihood
  kl_div_running = 0
  # Number of tokens processed to far
  tokens_processed = 0
  # Loop through each batch
  for i in range(0, num_samples, batch_size):
    torch.cuda.empty_cache()
    j = min(i + batch_size, num_samples)

    inputs = torch.cat(data[i:j]).to(device)
    targets = torch.cat(target_logits[i:j]).to(device)
    # Forward pass through the model
    lm_logits = model(inputs).logits

    # Don't predict last token (not required, can be removed)
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_targets = targets[:, :-1, :]

    # Squeeze on GPU
    torch.cuda.empty_cache()
    for i in range(0, shift_logits.shape[1], 1024):
      j = min(i + 1024, shift_logits.shape[1])
      shift_logits_batch = shift_logits[:, i:j, :]
      shift_targets_batch = shift_targets[:, i:j, :]
      loss_batch = torch.nn.functional.kl_div(
        shift_logits_batch.reshape(-1, shift_logits_batch.size(-1)).log_softmax(dim=-1),
        shift_targets_batch.reshape(-1, shift_targets_batch.size(-1)).log_softmax(
          dim=-1
        ),
        log_target=True,
        reduction="batchmean",
      )
      # Calculate negative log likelihood
      a = shift_targets_batch.numel() / (tokens_processed + shift_targets_batch.numel())
      b = tokens_processed / (tokens_processed + shift_targets_batch.numel())
      kl_div_running = a * loss_batch + b * kl_div_running
      # Update number of processed tokens
      tokens_processed += shift_targets_batch.numel()
      del shift_logits_batch, shift_targets_batch, loss_batch
      torch.cuda.empty_cache()

  return kl_div_running.item()


def compute_fitness(model, data, target_logits):
  return compute_kl_div(model, data, target_logits)


def selection(
  model,
  candidates,
  num_survive: int,
  calibration_data,
  num_tokens: int,
  target_logits: Optional[List[torch.Tensor]] = None,
):
  calibration_minibatch = []
  minibatch_ids = []
  target_logits_minibatch = []
  tokens_used = 0
  while tokens_used < num_tokens:  # generate minibatch with exactly num_tokens tokens
    minibatch_id = random.randint(0, len(calibration_data) - 1)
    if minibatch_id in minibatch_ids:  # avoid duplicates
      continue
    minibatch_ids.append(minibatch_id)
    if tokens_used + calibration_data[minibatch_id].shape[1] > num_tokens:
      calibration_minibatch.append(
        calibration_data[minibatch_id][:, : num_tokens - tokens_used]
      )
      target_logits_minibatch.append(
        target_logits[minibatch_id][:, : num_tokens - tokens_used]
      )
      tokens_used = num_tokens
    else:
      calibration_minibatch.append(calibration_data[minibatch_id])
      target_logits_minibatch.append(target_logits[minibatch_id])
      tokens_used += calibration_data[minibatch_id].shape[1]

  if len(target_logits_minibatch) == 0:
    target_logits_minibatch = None
  fitnesses = []
  for candidate in tqdm(candidates, desc="Fitness calculation"):
    maskModel(model, attnMask=candidate["attn"], mlpMask=candidate["mlp"])
    fitness = compute_fitness(model, calibration_minibatch, target_logits_minibatch)
    fitnesses.append(fitness)
    unmaskModel(model, attnMask=candidate["attn"], mlpMask=candidate["mlp"])
  # Keep only best
  best_ids = np.argsort(fitnesses)[:num_survive]
  return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids]


@torch.no_grad()
def evopress(model, num_prune, tokenizer, dataset_c4, drop_entire_block=False):
  # Default configuration used by Evopress for the depth pruning experiments,
  # taken from: https://github.com/IST-DASLab/EvoPress/blob/main/example_scripts/drop_search.sh
  args = {
    "calibration_tokens": 131072,
    "calibration_sequence_length": 8192,
    "offspring": 32,
    "population_size": 1,
    "initially_generated": 64,
    "initial_tokens": 2048,
    "survivors_per_selection": [2, 1],
    "tokens_per_selection": [2048, 32768],
    "max_mutations": 3,
  }

  args["calibration_sequence_length"] = (
    args["calibration_sequence_length"] or model.config.max_position_embeddings
  )

  if model.config.model_type in ("llama", "phi3"):
    args["calibration_sequence_length"] = 4096
  elif model.config.model_type == "qwen2":
    args["calibration_sequence_length"] = 2048

  layers = model.model.layers
  total_blocks = len(layers)
  num_generations = int(
    num_prune * (total_blocks - num_prune) / 1.5
  )  # Table 8 in the paper
  device = next(model.parameters()).device

  # Load calibration data

  calibration_data = collect_samples_with_join(
    dataset_c4,
    tokenizer,
    args["calibration_tokens"] // args["calibration_sequence_length"],
    args["calibration_sequence_length"],
  )

  target_logits = []
  # Compute target logits (calibration)
  for i in tqdm(range(len(calibration_data)), desc="Computing target logits (calib)"):
    target_logits.append(model(calibration_data[i].to(device)).logits.cpu())

  initial_population_candidates = []  # store initially generated search points (only take fittest for first population)

  while len(initial_population_candidates) < args["initially_generated"]:
    removed_state = {"attn": [0] * total_blocks, "mlp": [0] * total_blocks}

    attn_remove_ind = random.sample(list(range(total_blocks)), num_prune)
    for ind in attn_remove_ind:
      removed_state["attn"][ind] = 1

    mlp_remove_ind = random.sample(list(range(total_blocks)), num_prune)
    for ind in mlp_remove_ind:
      removed_state["mlp"][ind] = 1

    if drop_entire_block:
      removed_state["mlp"] = copy.deepcopy(removed_state["attn"])

    if removed_state in initial_population_candidates:  # avoid duplicates
      continue

    initial_population_candidates.append(removed_state)

  population, train_fitnesses = selection(
    model=model,
    candidates=initial_population_candidates,
    num_survive=args["population_size"],
    calibration_data=calibration_data,
    num_tokens=args["initial_tokens"],
    target_logits=target_logits,
  )

  best_individual = None
  for gen_id in range(num_generations):
    print(f"Generation {gen_id + 1}/{num_generations}")
    print(f"Train fitness {train_fitnesses[0]:.2e}")

    for parent in population:
      print(
        f"Parent: attn: {[int(ele) for ele in parent['attn']]} mlp: {[int(ele) for ele in parent['mlp']]}"
      )

    offspring_list = []

    # Generate offspring by Mutation
    while len(offspring_list) < args["offspring"]:
      offspring = copy.deepcopy(random.choice(population))

      # Mutation
      num_flips = min(
        random.randint(1, args["max_mutations"]),
        random.randint(1, args["max_mutations"]),
      )  # bias towards lower values
      for _ in range(num_flips):
        remove_type = random.randint(0, 1)  # 0 remove attention, 1 remove mlp
        if remove_type == 0:
          subblock_type = "attn"
        else:
          subblock_type = "mlp"

        remove_ind = random.randint(0, total_blocks - 1)
        while offspring[subblock_type][remove_ind] == 1:
          remove_ind = random.randint(0, total_blocks - 1)

        add_ind = random.randint(0, total_blocks - 1)
        while offspring[subblock_type][add_ind] == 0:
          add_ind = random.randint(0, total_blocks - 1)

        offspring[subblock_type][remove_ind] = 1
        offspring[subblock_type][add_ind] = 0

      # Check if pruning an entire block, if attn submodule is
      # pruned then the mlp is pruned as well
      if drop_entire_block:
        offspring["mlp"] = copy.deepcopy(offspring["attn"])

      if offspring in offspring_list or offspring in population:  # avoid duplicates
        continue

      offspring_list.append(offspring)

    # Selection in multiple steps
    for num_survive, num_tokens in zip(
      args["survivors_per_selection"], args["tokens_per_selection"]
    ):
      if num_survive == args["survivors_per_selection"][-1]:
        for i in range(
          len(population)
        ):  # Elitist EA: Add search points in current generation to final selection step
          if population[i] not in offspring_list:
            offspring_list.append(population[i])

      offspring_list, train_fitnesses = selection(
        model=model,
        candidates=offspring_list,
        num_survive=num_survive,
        calibration_data=calibration_data,
        num_tokens=num_tokens,
        target_logits=target_logits,
      )

    population = offspring_list
    best_individual = population[0]

  if drop_entire_block:
    return best_individual["attn"]
  else:
    return best_individual["attn"], best_individual["mlp"]