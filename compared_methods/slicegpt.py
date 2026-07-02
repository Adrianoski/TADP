# The following code was sourced and adapted from the repository:
# https://github.com/microsoft/TransformerCompression

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from .slicegpt_utils import hf_utils, layernorm_fusion, rotate
from .slicegpt_utils.slicing_scheduler import ConstSlicingScheduler
from .slicegpt_utils.gpu_utils import distribute_model

@torch.no_grad()
def slicegpt(model_path, sparsity, calibration_dataset):
  
  if "llama" in model_path.lower() or "phi-3" in model_path.lower() or "qwen2" in model_path.lower():
    dtype = torch.bfloat16
  else:
    dtype = torch.float16

  if "llama-2-7b-hf" in model_path.lower():
    model_name = "meta-llama/Llama-2-7b-hf"
  elif "llama-3.2-3b" in model_path.lower():
    model_name = "meta-llama/Llama-3.2-3B"
  elif "meta-llama-3-8b" in model_path.lower() or "meta_llama_3_8b" in model_path.lower():
    model_name = "meta-llama/Meta-Llama-3-8B"
  elif "mistral-7b-v0.1" in model_path.lower():
    model_name = "mistralai/Mistral-7B-v0.1"
  elif "mistral-7b-v0.3" in model_path.lower():
    model_name = "mistralai/Mistral-7B-v0.3"
  elif "qwen2.5-3b" in model_path.lower():
    model_name = "Qwen/Qwen2.5-3B"
  elif "qwen2.5-7b" in model_path.lower():
    model_name = "Qwen/Qwen2.5-7B"
  elif "phi-3" in model_path.lower():
    model_name = "microsoft/Phi-3-medium-128k-instruct"
  else:
    print(f"ERROR: model {model_path} is not supported")
    exit(1)

  # load one of the pre-trained models
  model_adapter, _ = hf_utils.get_model_and_tokenizer(
    model_name, model_path, dtype=dtype
  )
  
  # replace modules with compressible equivalents
  layernorm_fusion.replace_layers(model_adapter)

  # fuse layernorms and add rotations to skip connections
  layernorm_fusion.fuse_modules(model_adapter)

  if "llama" in model_path.lower() or "mistral" in model_path.lower():
    distribute_model(model_adapter)

  # compute new embedding dimension given the desired sparsity level
  new_embedding_dimension = int((1 - sparsity) * model_adapter.hidden_size)
  # round (down) to the nearest multiple of round_interval
  new_embedding_dimension -= new_embedding_dimension % 8
  print(
      f"New embedding dimension: {new_embedding_dimension} (sparsity {100*(1 - new_embedding_dimension / model_adapter.hidden_size):.4f} %)"
  )

  scheduler = ConstSlicingScheduler(new_embedding_dimension)
  rotate.rotate_and_slice(model_adapter, calibration_dataset, scheduler, final_orientation="random", apply_mask=False)

  distribute_model(model_adapter)
  
  return model_adapter.model
