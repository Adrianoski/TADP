import os
import logging
import torch
import lm_eval
from tqdm import tqdm
from datetime import datetime
from lm_eval import tasks as lm_tasks, evaluator as lm_evaluator
from lm_eval.models.huggingface import HFLM


# Perplexity evaluation on all the three different datasets
def evaluation_ppl(model, wikitext_input_ids, c4_val_input_ids, fineweb_edu_input_ids):
  ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=512)
  logging.info(f"Perplexity (wikitext2): {ppl}")

  ppl = evaluate_perplexity(model, c4_val_input_ids, seq_len=512)
  logging.info(f"Perplexity (c4): {ppl}")

  ppl = evaluate_perplexity(model, fineweb_edu_input_ids, seq_len=512)
  logging.info(f"Perplexity (fineweb-edu): {ppl}")



# The following code was sourced and adapted from the repository:
# https://github.com/IST-DASLab/EvoPress
@torch.no_grad()
def evaluate_perplexity(
  model, input_ids, seq_len=512, batch_size=1, enable_tqdm=True, device="cuda"
):
  # Divide the test dataset into samples of exactly seq_len tokens
  num_samples = input_ids.numel() // seq_len
  data = []
  for i in range(num_samples):
    data.append(input_ids[:, i * seq_len : (i + 1) * seq_len])

  # Running estimate of negative log-likelihood
  nll_running = 0
  tokens_processed = 0

  if enable_tqdm:
    ppl_range = tqdm(range(0, num_samples, batch_size), desc=f"Calculating perplexity")
  else:
    ppl_range = range(0, num_samples, batch_size)

  # Loop through each batch
  for i in ppl_range:
    j = min(i + batch_size, num_samples)
    inputs = torch.cat(data[i:j]).to(device)
    # Forward pass through the model
    lm_logits = model(inputs).logits
    # Shift logits and labels for next token prediction
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = inputs[:, 1:]
    # Compute loss
    loss = torch.nn.functional.cross_entropy(
      shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
    )
    # Calculate negative log likelihood
    a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
    b = tokens_processed / (tokens_processed + shift_labels.numel())
    nll_running = a * loss + b * nll_running
    # Update number of processed tokens
    tokens_processed += shift_labels.numel()

  # Compute perplexity
  ppl = nll_running.exp().item()
  return ppl



# Downstream tasks evaluation with Lm Eval Harness
def evaluation_downstream(model, model_name, num_fewshot=0):
  task_list = ["winogrande", "arc_easy", "arc_challenge", "hellaswag", "piqa", "mmlu"]
  results = eval_zero_shot(
    model_name,
    model,
    task_list,
    num_fewshot=num_fewshot,
    use_accelerate=True,
    access_token=os.environ.get("HF_TOKEN"),
  )

  logging.info("Zero-shot evaluation results")
  for task in results["results"].keys():
    task_res = results["results"][task]
    logging.info(f"{task_res['alias']} : {task_res['acc,none']}")


# The following code was sourced and adapted from the repository:
# https://github.com/eliacunegatti/NeuroAL
def eval_zero_shot(
  model_name,
  model,
  task_list=["arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande"],
  num_fewshot=0,
  use_accelerate=False,
  access_token=None,
):
  model_args = f"token={access_token}"
  limit = None
  if "70b" in model_name or "65b" in model_name:
    limit = 2000
  if use_accelerate:
    model_args = f"use_accelerate=True"

  logging.info(f"Testing tasks: {task_list}")
  model_obj = lm_eval.models.huggingface.HFLM(pretrained=model)
  results = lm_eval.evaluator.simple_evaluate(
    model=model_obj,
    model_args=model_args,
    tasks=task_list,
    num_fewshot=num_fewshot,
    batch_size=None,
    device=None,
    limit=limit,
    check_integrity=False,
  )

  return results


@torch.no_grad()
def evaluate_inference_time(model, sample):
  sample = sample.to("cuda")

  # model warmup
  for _ in range(10):
    _ = model(sample)

  # Cuda events counter
  n_runs = 10
  times = []
  for _ in range(n_runs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    _ = model(sample)
    torch.cuda.synchronize()
    end.record()

    torch.cuda.synchronize()

    elapsed_time = start.elapsed_time(end)
    logging.info(f"Inference time: {elapsed_time} s")
    times.append(elapsed_time)

  average_time = sum(times) / n_runs
  logging.info(f"Average Inference Time: {average_time:.6f} seconds")


@torch.no_grad()
def generate_response(prompt, model, tokenizer, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    generation_kwargs = {
        "max_length": max_length,
        "num_beams": 5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        if tokenizer.pad_token_id != tokenizer.eos_token_id:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    
    outputs = model.generate(
        **inputs,
        **generation_kwargs
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@torch.no_grad()
def qualitative_results(model, tokenizer, max_length=64):
  prompt1 = "Who is Albert Einstein?"
  prompt2 = "The theory of relativity"

  logging.info("Prompt 1: Who is Albert Einstein?")
  logging.info("-" * 20)
  logging.info(generate_response(prompt1, model, tokenizer, max_length))
  logging.info("Prompt 2: Explain the theory of relativity")
  logging.info("-" * 20)
  logging.info(generate_response(prompt2, model, tokenizer, max_length))


# ==================== OUR METHODOLOGY EVALUATION ====================

def evaluate_downstream_our(model, tokenizer, task_list, device="cuda"):
  """Downstream benchmark evaluation via lm_eval (our methodology)."""
  logging.info(f"[Our Methodology] Running lm_eval tasks: {task_list}")
  lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device, dtype="float16")
  task_manager = lm_tasks.TaskManager()
  task_dict = lm_evaluator.get_task_dict(task_list, task_manager)
  results = lm_evaluator.evaluate(lm=lm, task_dict=task_dict)
  scores = {}
  for task, res in results["results"].items():
    acc = res.get("acc_norm,none", res.get("acc,none", None))
    if acc is not None:
      scores[task] = round(acc * 100, 2)
      logging.info(f"  {task}: {scores[task]:.2f}%")
  return scores


def save_results(results_file, method_name, sparsity, ppl, params,
                 generation, baseline_params, baseline_ppl, downstream=None):
  """Save evaluation results in the format of our methodology."""
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  os.makedirs(os.path.dirname(results_file) if os.path.dirname(results_file) else '.', exist_ok=True)
  with open(results_file, 'a') as f:
    f.write("=" * 80 + "\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Method: {method_name}\n")
    f.write(f"Sparsity: {sparsity * 100:.1f}%\n")
    f.write("-" * 80 + "\n")
    f.write(f"Parameters: {params:,}\n")
    f.write(f"Param Reduction: {(1 - params / baseline_params) * 100:.2f}%\n")
    f.write(f"Perplexity: {ppl:.4f}\n")
    f.write(f"PPL Change vs Baseline: {(ppl / baseline_ppl - 1) * 100:+.2f}%\n")
    if downstream:
      f.write("Downstream Tasks:\n")
      for task, acc in downstream.items():
        f.write(f"  {task}: {acc:.2f}%\n")
    f.write(f"Generation: {generation}\n")
    f.write("=" * 80 + "\n\n")