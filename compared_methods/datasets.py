from datasets import load_dataset, load_from_disk
import logging

def load_wikitext2(local=False):
  if local:
    dataset = load_from_disk("data/wikitext-2")
  else:
    try:
      dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except:
      logging.error(
        "Unable to download Wikitext-2 dataset from huggingface. Falling back to local version"
      )
      dataset = load_from_disk("data/wikitext-2")

  return dataset


def load_c4(train, local=False):
  if train:
    if local:
      dataset = load_from_disk("data/c4_train")
    else:
      try:
        dataset = load_dataset(
          "allenai/c4",
          "default",
          data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
          split="train[:1000]",
          revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
      except:
        logging.error(
          "Unable to download C4 dataset from huggingface. Falling back to local version"
        )
        dataset = load_from_disk("data/c4_train")
  else:
    if local:
      dataset = load_from_disk("data/c4_val")
    else:
      try:
        dataset = load_dataset(
          "allenai/c4",
          "default",
          data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
          split="validation[:1100]",
          revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
      except:
        logging.error(
          "Unable to download C4 dataset from huggingface. Falling back to local version"
        )
        dataset = load_from_disk("data/c4_val")

  return dataset


def load_fineweb_edu(local=False):
  if local:
    dataset = load_from_disk("data/fineweb-edu")
  else:
    try:
      dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train[:1100]",
        data_files=["sample/10BT/000_00000.parquet"],
      )
    except:
      logging.error(
        "Unable to download fineweb-edu dataset from huggingface. Falling back to local version"
      )
      dataset = load_from_disk("data/fineweb-edu")

  return dataset
