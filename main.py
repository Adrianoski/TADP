import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from config import (
    device,
    MODEL_NAME,
    RECOVERY_SAMPLES,
    MAX_LENGTH,
    BATCH_SIZE,
    PRUNE_PERCENT,
    TADP_CONFIG,
)
from utils import count_parameters, generate_text
from data import prepare_dataset
from evaluation import evaluate_perplexity
from temporal_pruning import TemporalPruning
from pruning import apply_pruning


def main():
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Configuration loaded.")
    print(f"TADP Config: {TADP_CONFIG}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    wiki_data = load_dataset(
        'wikitext', 'wikitext-2-raw-v1', split=f'train[:{RECOVERY_SAMPLES}]'
    )
    wiki_loader = prepare_dataset(
        wiki_data, tokenizer, max_length=MAX_LENGTH, batch_size=BATCH_SIZE
    )
    print(f"WikiText samples: {len(wiki_data)}")

    test_prompt = "Paris is the capital of"

    print("=" * 70)
    print("BASELINE MODEL")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print("TP - TEMPORAL PRUNING METHOD")
    print("=" * 70)
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
    print(f"\nParameters: {TADP_params:,} "
          f"({(1 - TADP_params / base_params) * 100:.1f}% reduction)")

    TADP_metrics = evaluate_perplexity(TADP_model, wiki_loader, device)
    print(f"Perplexity (Wiki): {TADP_metrics['perplexity']:.2f} "
          f"(+{(TADP_metrics['perplexity'] / base_metrics['perplexity'] - 1) * 100:.1f}%)")

    TADP_gen = generate_text(TADP_model, tokenizer, test_prompt, device)
    print(f"Generation: {TADP_gen}")


if __name__ == '__main__':
    main()
