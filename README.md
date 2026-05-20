# TADP — Temporal Activation-Dynamics Pruning

Structured pruning method for LLMs based on temporal activation dynamics.
Each MLP neuron is scored by a hybrid combination of:

- **Mean activation** — frequent-pattern signal
- **Peak activation** — selective neuron preservation
- **Temporal variance** — context sensitivity across token positions
- **Temporal gradient** — transition dynamics between consecutive positions
- **Linearly position-weighted activation** — positional importance bias (weights from 0.5 to 1.5)

The lowest-scoring neurons are removed structurally from `gate_proj`,
`up_proj` and `down_proj` of each transformer layer.

## Project layout

```
TADP/
├── config.py             # constants (model, dataset, pruning ratio, TADP config)
├── utils.py              # GPU cache, parameter count, text generation
├── data.py               # dataset tokenization and DataLoader
├── evaluation.py         # perplexity computation
├── temporal_pruning.py   # TemporalPruning class (importance scoring)
├── pruning.py            # structural pruning operations on MLP
└── main.py               # entry point: baseline + pruning + evaluation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Edit `config.py` to change model, calibration samples, sequence length,
batch size, or pruning ratio.

## Configuration

| Variable           | Default                       | Description                         |
| ------------------ | ----------------------------- | ----------------------------------- |
| `MODEL_NAME`       | `meta-llama/Llama-3.2-3B`     | HuggingFace model identifier        |
| `RECOVERY_SAMPLES` | `100`                         | Calibration samples from WikiText-2 |
| `MAX_LENGTH`       | `512`                         | Sequence length                     |
| `BATCH_SIZE`       | `4`                           | Calibration / eval batch size       |
| `PRUNE_PERCENT`    | `0.4`                         | Fraction of MLP neurons removed     |
| `TADP_CONFIG`      | `temporal_weight=1.0`         | Weight applied to the temporal score |
