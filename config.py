import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = 'meta-llama/Llama-3.2-3B'

RECOVERY_SAMPLES = 100
MAX_LENGTH = 512
BATCH_SIZE = 4

PRUNE_PERCENT = 0.4

TADP_CONFIG = {
    'temporal_weight': 1.0,
}
