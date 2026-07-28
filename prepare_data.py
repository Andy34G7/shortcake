import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
from datasets import load_dataset

DATASETS_CONFIG = [
    {"name": "HuggingFaceTB/SmolLM-Corpus", "config": "cosmopedia-v2", "column": "text", "max_samples": 60000},
    {"name": "HuggingFaceTB/SmolLM-Corpus", "config": "fineweb-edu-dedup", "column": "text", "max_samples": 60000},
    {"name": "HuggingFaceTB/SmolLM-Corpus", "config": "python-edu", "column": "content", "max_samples": 40000},
    {"name": "codeparrot/codeparrot-clean", "config": None, "column": "content", "max_samples": 40000},
]



def prepare_bin_data(
    tokenizer_path: str = "tokenizer.json",
    output_dir: str = "data",
    val_ratio: float = 0.05,
    test_ratio: float = 0.02,
):
    """Streams the 3 hardcoded code datasets, tokenizes them, and saves uint16 binary files."""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run train_tokenizer.py first!")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    os.makedirs(output_dir, exist_ok=True)

    token_list = []
    total_docs = 0

    for target in DATASETS_CONFIG:
        ds_name = target["name"]
        ds_cfg = target["config"]
        col = target["column"]
        max_s = target["max_samples"]

        print(f"\n--- Streaming up to {max_s:,} documents from '{ds_name}' (config: {ds_cfg}) ---")
        kwargs = {}
        if ds_cfg:
            kwargs["name"] = ds_cfg

        try:
            ds = load_dataset(ds_name, split="train", streaming=True, **kwargs)
            count = 0
            for sample in tqdm(ds, total=max_s, desc=ds_name):
                text = sample.get(col, "")
                if text and len(text.strip()) > 0:
                    encoding = tokenizer.encode(text)
                    token_list.extend(encoding.ids)
                    count += 1
                    total_docs += 1
                    if count >= max_s:
                        break
        except Exception as e:
            print(f"Warning: Could not stream dataset '{ds_name}': {e}")

    token_ids = np.array(token_list, dtype=np.uint16)
    n_tokens = len(token_ids)
    print(f"\n==========================================")
    print(f"Total tokens collected from {total_docs:,} documents: {n_tokens:,}")
    print(f"==========================================")

    # Split into train / val / test
    n_val = int(n_tokens * val_ratio)
    n_test = int(n_tokens * test_ratio)
    n_train = n_tokens - n_val - n_test

    train_ids = token_ids[:n_train]
    val_ids = token_ids[n_train : n_train + n_val]
    test_ids = token_ids[n_train + n_val :]

    train_filename = os.path.join(output_dir, "train.bin")
    val_filename = os.path.join(output_dir, "val.bin")
    test_filename = os.path.join(output_dir, "test.bin")

    train_ids.tofile(train_filename)
    val_ids.tofile(val_filename)
    test_ids.tofile(test_filename)

    print(f" Saved {len(train_ids):,} train tokens to {train_filename}")
    print(f" Saved {len(val_ids):,} validation tokens to {val_filename}")
    print(f" Saved {len(test_ids):,} test tokens to {test_filename}")


if __name__ == "__main__":
    prepare_bin_data()
