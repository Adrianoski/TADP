import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch import nn
import os
import json
import copy
import gc
import time
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from optipfair import prune_model

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

MODEL_NAME = 'meta-llama/Llama-3.2-3B'
RECOVERY_SAMPLES = 100
MAX_LENGTH = 512
BATCH_SIZE = 4

PRUNE_PERCENT = 0.4

TADP_CONFIG = {
    'temporal_weight': 1.0,
    'position_bias': 'linear',
}

print("Configuration loaded.")
print(f"TADP Config: {TADP_CONFIG}")

def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def prepare_dataset(dataset, tokenizer, text_field='text', max_length=512, batch_size=4):
    def tokenize_function(examples):
        if text_field in examples:
            texts = examples[text_field]
        elif 'sms' in examples:
            texts = examples['sms']
        else:
            texts = examples[list(examples.keys())[0]]

        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False)

def evaluate_perplexity(model, dataloader, device):
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            num_real_tokens = attention_mask.sum().item()

            total_loss += outputs.loss.item() * num_real_tokens
            total_tokens += num_real_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return {
        'loss': avg_loss,
        'perplexity': perplexity
    }


def generate_text(model, tokenizer, prompt, device, max_new_tokens=50):
    model.eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

class TemporalPruning:
    """
    Temporal Pruning (TP)

    Temporal Dynamics: Variance and gradient of activations across positions
    """

    def __init__(self, model, device, config=None):
        self.model = model
        self.device = device
        self.config = config or TADP_CONFIG
        self._temporal_activations = {}

    def setup_hooks(self):
        self._temporal_activations.clear()
        handles = []

        for idx, layer in enumerate(self.model.model.layers):
            self._temporal_activations[idx] = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                X_d = input[0].detach().float()
                pos_norms = torch.norm(X_d, p=2, dim=0)
                self._temporal_activations[layer_idx].append(pos_norms.cpu())
            return hook

        for idx, layer in enumerate(self.model.model.layers):
            handle = layer.mlp.down_proj.register_forward_hook(make_hook(idx))
            handles.append(handle)

        return handles

    def calibrate(self, dataloader):
        handles = self.setup_hooks()
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="TADP Calibration"):
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                self.model(**inputs)

        for h in handles:
            h.remove()

        processed_data = {}
        for layer_idx in self._temporal_activations:
            temporal_stack = torch.stack(self._temporal_activations[layer_idx])
            processed_data[layer_idx] = {'temporal': temporal_stack}

        return processed_data

    def _compute_temporal_score(self, layer_data):
        """
        Temporal dynamics score with peak-activation preservation.

        Components:
        - mean_act:      average activation norm (common pattern signal)
        - peak_act:      max activation across batches (selective neuron preservation)
        - temporal_var:  variance across sequence positions (context sensitivity)
        - temporal_grad: gradient between consecutive positions (transition dynamics)
        - weighted_act:  position-weighted activation (positional importance)
        """
        temporal = layer_data['temporal']
        avg_temporal = temporal.mean(dim=0)
        seq_len = avg_temporal.size(0)

        mean_act = torch.norm(avg_temporal, p=2, dim=0)

        max_temporal = temporal.max(dim=0).values
        peak_act = torch.norm(max_temporal, p=2, dim=0)

        temporal_var = avg_temporal.var(dim=0)

        if seq_len > 1:
            gradients = torch.abs(avg_temporal[1:] - avg_temporal[:-1])
            mean_gradient = gradients.mean(dim=0)
        else:
            mean_gradient = torch.zeros_like(mean_act)

        if self.config['position_bias'] == 'linear':
            position_weights = torch.linspace(0.5, 1.5, seq_len)
        elif self.config['position_bias'] == 'exponential':
            position_weights = torch.exp(torch.linspace(0, 1, seq_len))
        else:
            position_weights = torch.ones(seq_len)

        position_weights = position_weights.unsqueeze(1)
        weighted_act = (avg_temporal * position_weights).sum(dim=0) / position_weights.sum()

        mean_norm     = mean_act      / (mean_act.max()      + 1e-8)
        peak_norm     = peak_act      / (peak_act.max()      + 1e-8)
        var_norm      = temporal_var  / (temporal_var.max()   + 1e-8)
        grad_norm     = mean_gradient / (mean_gradient.max()  + 1e-8)
        weighted_norm = weighted_act  / (weighted_act.max()   + 1e-8)

        return (0.15 * mean_norm +
                0.35 * peak_norm +
                0.20 * var_norm  +
                0.15 * grad_norm +
                0.15 * weighted_norm)

    def compute_importance(self, layer_idx, calibration_data):
        layer_data = calibration_data[layer_idx]
        temporal_score = self._compute_temporal_score(layer_data)
        importance = self.config['temporal_weight'] * temporal_score
        return importance.to(self.device)

def prune_neuron_pairs(mlp, prune_percent, importance_scores, layer_idx, device):
    gate_weight = mlp.gate_proj.weight.data
    up_weight = mlp.up_proj.weight.data
    down_weight = mlp.down_proj.weight.data

    original_size = gate_weight.size(0)
    num_to_prune = min(int(prune_percent * original_size), original_size - 1)
    k = original_size - num_to_prune

    _, indices_to_keep = torch.topk(importance_scores, k, largest=True, sorted=True)
    indices_to_keep = indices_to_keep.sort().values

    new_gate = nn.Linear(mlp.gate_proj.in_features, k, bias=False).to(device)
    new_up = nn.Linear(mlp.up_proj.in_features, k, bias=False).to(device)
    new_down = nn.Linear(k, mlp.down_proj.out_features, bias=False).to(device)

    new_gate.weight.data = gate_weight[indices_to_keep, :]
    new_up.weight.data = up_weight[indices_to_keep, :]
    new_down.weight.data = down_weight[:, indices_to_keep]

    return new_gate, new_up, new_down, k


def apply_pruning(model, prune_percent, importance_computer, calibration_data, device, method_name):
    print(f"\n{'='*60}")
    print(f"Applying {method_name} with {prune_percent*100:.1f}% pruning")
    print(f"{'='*60}\n")

    new_intermediate_size = None

    for idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        original_size = mlp.gate_proj.out_features

        importance = importance_computer.compute_importance(idx, calibration_data)
        new_gate, new_up, new_down, new_size = prune_neuron_pairs(
            mlp, prune_percent, importance, idx, device
        )

        mlp.gate_proj = new_gate
        mlp.up_proj = new_up
        mlp.down_proj = new_down

        if new_intermediate_size is None:
            new_intermediate_size = new_size

        if (idx + 1) % 4 == 0:
            print(f"  Layers {idx-3:2d}-{idx:2d}: {original_size} -> {new_size} ({new_size/original_size*100:.1f}% kept)")

    model.config.intermediate_size = new_intermediate_size
    print(f"\n✓ Pruning complete. New intermediate size: {new_intermediate_size}")

    return model

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    wiki_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=f'train[:{RECOVERY_SAMPLES}]')
    wiki_loader = prepare_dataset(wiki_data, tokenizer, max_length=MAX_LENGTH, batch_size=BATCH_SIZE)
    print(f"✓ WikiText samples: {len(wiki_data)}")

    test_prompt = "Paris is the capital of"

    print("="*70)
    print("BASELINE MODEL")
    print("="*70)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    base_model.eval()

    base_params = count_parameters(base_model)
    print(f"Parameters: {base_params:,}")

    base_metrics = evaluate_perplexity(base_model, wiki_loader, device)
    print(f"Perplexity (Wiki): {base_metrics['perplexity']:.2f}")

    base_gen = generate_text(base_model, tokenizer, test_prompt, device)
    print(f"Generation: {base_gen}")

    print("\n" + "="*70)
    print("TP - TEMPORAL PRUNING METHOD")
    print("="*70)
    print(f"Config: {TADP_CONFIG}")

    TADP_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )

    TADP_pruner = TemporalPruning(TADP_model, device, TADP_CONFIG)
    TADP_calibration_data = TADP_pruner.calibrate(wiki_loader)

    TADP_model = apply_pruning(
        TADP_model, PRUNE_PERCENT, TADP_pruner, TADP_calibration_data, device, "TADP"
    )

    TADP_params = count_parameters(TADP_model)
    print(f"\nParameters: {TADP_params:,} ({(1-TADP_params/base_params)*100:.1f}% reduction)")

    TADP_metrics = evaluate_perplexity(TADP_model, wiki_loader, device)
    print(f"Perplexity (Wiki): {TADP_metrics['perplexity']:.2f} (+{(TADP_metrics['perplexity']/base_metrics['perplexity']-1)*100:.1f}%)")

    TADP_gen = generate_text(TADP_model, tokenizer, test_prompt, device)
    print(f"Generation: {TADP_gen}")
