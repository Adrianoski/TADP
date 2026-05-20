import torch
from tqdm import tqdm

from config import TADP_CONFIG


class TemporalPruning:
    """
    Temporal Pruning (TP)

    Temporal Dynamics: Variance and gradient of activations across positions.
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
                    'attention_mask': batch['attention_mask'].to(self.device),
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
        - weighted_act:  linearly position-weighted activation (positional importance)
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

        position_weights = torch.linspace(0.5, 1.5, seq_len).unsqueeze(1)
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
