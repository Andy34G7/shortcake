"""Build the small ablation corpus and tokenizer.

Collects Python source files from the local Python standard library (~24M characters),
trains a 4,096-token byte-level BPE tokenizer with the repo's
`train_bpe_tokenizer_from_iterator`, and writes uint16 token streams:

    ablations/out/tokenizer.json
    ablations/out/train.bin
    ablations/out/val.bin

Run from the repo root:  python ablations/prep.py      (takes about a minute)

The corpus depends on the installed Python version, so token counts (and losses) are only
comparable between runs made against the same interpreter.
"""

import argparse
import os
import random
import sys
import sysconfig
import types
from pathlib import Path

import numpy as np

ABLATIONS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ABLATIONS_DIR.parent
DEFAULT_OUT_DIR = ABLATIONS_DIR / "out"
sys.path.insert(0, str(REPO_ROOT))

# train_tokenizer.py does `from datasets import load_dataset` at import time, but the function
# used here never touches it. Only stub the module if the real package is not installed.
try:
    import datasets  # noqa: F401
except ImportError:
    _stub = types.ModuleType("datasets")
    _stub.load_dataset = None
    sys.modules["datasets"] = _stub

from tokenizers import Tokenizer  # noqa: E402
from train_tokenizer import train_bpe_tokenizer_from_iterator  # noqa: E402

MIN_FILE_BYTES = 2_000      # skip tiny files (mostly __init__.py stubs)
MAX_FILE_BYTES = 200_000    # skip huge generated files
TARGET_CHARS = 24_000_000
VAL_DIVISOR = 20            # 1/20 of the files (split by file) go to validation
SEED = 0


def collect_stdlib_sources(target_chars: int, seed: int) -> list:
    stdlib = sysconfig.get_paths()["stdlib"]
    paths = []
    for dirpath, _, filenames in os.walk(stdlib):
        for name in filenames:
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                if MIN_FILE_BYTES < os.path.getsize(path) < MAX_FILE_BYTES:
                    paths.append(path)
    # Shuffled in os.walk order (not sorted) to reproduce the corpus behind the README numbers;
    # os.walk order is filesystem dependent, so another machine may pick different files.
    random.Random(seed).shuffle(paths)

    texts, total = [], 0
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        texts.append(text)
        total += len(text)
        if total > target_chars:
            break
    print(f"Collected {len(texts)} files, {total / 1e6:.1f}M characters from {stdlib}")
    return texts


def encode(tokenizer: Tokenizer, texts: list) -> np.ndarray:
    ids = []
    for enc in tokenizer.encode_batch(texts):
        ids.extend(enc.ids)
    return np.array(ids, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--chars", type=int, default=TARGET_CHARS, help="approximate corpus size in characters")
    args = ap.parse_args()
    assert args.vocab_size <= 65536, "token ids are stored as uint16"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = args.out_dir / "tokenizer.json"

    texts = collect_stdlib_sources(args.chars, SEED)
    train_bpe_tokenizer_from_iterator(texts, output_path=str(tok_path), vocab_size=args.vocab_size)
    tokenizer = Tokenizer.from_file(str(tok_path))

    n_val = max(1, len(texts) // VAL_DIVISOR)
    val_ids = encode(tokenizer, texts[:n_val])
    train_ids = encode(tokenizer, texts[n_val:])
    val_ids.tofile(args.out_dir / "val.bin")
    train_ids.tofile(args.out_dir / "train.bin")
    print(f"train tokens {len(train_ids):,}  val tokens {len(val_ids):,}  -> {args.out_dir}")


if __name__ == "__main__":
    main()
