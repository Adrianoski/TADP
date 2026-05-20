import torch
from torch import nn


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
    print(f"\nPruning complete. New intermediate size: {new_intermediate_size}")

    return model
