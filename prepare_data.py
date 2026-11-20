import os
import json
import time
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
from tokenizers import Tokenizer
from tqdm import tqdm
from datasets import load_dataset


DATASETS_CONFIG: List[Dict[str, Any]] = [
    {"name": "HuggingFaceTB/SmolLM-Corpus", "config": "fineweb-edu-dedup", "column": "text", "weight": 0.45},
    {"name": "HuggingFaceTB/SmolLM-Corpus", "config": "cosmopedia-v2", "column": "text", "weight": 0.30},
    {"name": "codeparrot/codeparrot-clean", "config": None, "column": "content", "weight": 0.25},
]


class ShardWriter:
    """Streams uint16 token arrays into fixed-size binary shards on disk with bounded RAM."""

    def __init__(self, output_dir: str, prefix: str = "train", shard_size: int = 100_000_000):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.pos = 0
        self.shard_idx = 0
        self.total_tokens = 0
        self.shard_files: List[str] = []

    def add(self, tokens: np.ndarray):
        n = len(tokens)
        offset = 0
        while offset < n:
            space = self.shard_size - self.pos
            take = min(space, n - offset)
            self.buffer[self.pos : self.pos + take] = tokens[offset : offset + take]
            self.pos += take
            offset += take
            self.total_tokens += take
            if self.pos == self.shard_size:
                self._flush()

    def _flush(self):
        if self.pos == 0:
            return
        filename = f"{self.prefix}_{self.shard_idx:04d}.bin"
        filepath = os.path.join(self.output_dir, filename)
        self.buffer[: self.pos].tofile(filepath)
        self.shard_files.append(filename)
        print(f"  [Shard] Wrote {self.pos:,} tokens -> {filepath}", flush=True)
        self.shard_idx += 1
        self.pos = 0

    def finalize(self) -> List[str]:
        self._flush()
        return self.shard_files


def _open_stream(target: Dict[str, Any]):
    kwargs = {}
    if target.get("config"):
        kwargs["name"] = target["config"]
    ds = load_dataset(target["name"], split="train", streaming=True, **kwargs)
    return iter(ds)


def prepare_bin_data(
    tokenizer_path: str = "tokenizer.json",
    output_dir: str = "data",
    target_tokens: int = 1_000_000_000,
    shard_size: int = 100_000_000,
    batch_docs: int = 1024,
    val_ratio: float = 0.01,
    test_ratio: float = 0.005,
    max_val_tokens: int = 20_000_000,
    max_test_tokens: int = 10_000_000,
    seed: int = 42,
):
    """Weighted-interleaved streaming with batched Rust tokenization into sharded uint16 .bin files."""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run train_tokenizer.py first!")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer is missing '<eos>' token. Retrain tokenizer with train_tokenizer.py.")

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Open streaming iterators and validate that the expected text column exists
    streams = []
    weights = []
    for target in DATASETS_CONFIG:
        label = f"{target['name']} ({target['config']})" if target.get("config") else target["name"]
        try:
            it = _open_stream(target)
            first_row = next(it)
            if target["column"] not in first_row:
                print(
                    f"Warning: Skipping '{label}' — column '{target['column']}' not found in {list(first_row.keys())}",
                    flush=True,
                )
                continue
            streams.append({"target": target, "iter": it, "first_row": first_row, "label": label, "docs": 0})
            weights.append(float(target["weight"]))
            print(f"Opened stream: {label} (weight={target['weight']:.2f})", flush=True)
        except Exception as e:
            print(f"Warning: Could not open stream '{label}': {e}", flush=True)

    if not streams:
        raise RuntimeError("No dataset streams could be opened.")

    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    train_writer = ShardWriter(output_dir, prefix="train", shard_size=shard_size)
    val_chunks: List[np.ndarray] = []
    test_chunks: List[np.ndarray] = []
    val_tokens = 0
    test_tokens = 0

    target_val = min(max_val_tokens, max(500_000, int(target_tokens * val_ratio)))
    target_test = min(max_test_tokens, max(250_000, int(target_tokens * test_ratio)))

    t0 = time.time()
    total_docs = 0
    next_log_tokens = 25_000_000
    pbar = tqdm(total=target_tokens, unit="tok", unit_scale=True, desc="Tokenizing Shards")

    while train_writer.total_tokens < target_tokens and streams:
        # Pick a dataset stream according to configured weights
        idx = int(rng.choice(len(streams), p=weights))
        st = streams[idx]
        col = st["target"]["column"]

        # Pull a batch of documents for parallel Rust encode_batch
        texts: List[str] = []
        exhausted = False
        empty_streak = 0
        if st.get("first_row") is not None:
            first_text = st.pop("first_row").get(col, "")
            if first_text and len(first_text.strip()) > 0:
                texts.append(first_text)

        while len(texts) < batch_docs:
            try:
                sample = next(st["iter"])
            except StopIteration:
                exhausted = True
                break
            except Exception as e:
                print(f"Warning: Stream error on '{st['label']}': {e}", flush=True)
                exhausted = True
                break

            text = sample.get(col, "")
            if text and len(text.strip()) > 0:
                texts.append(text)
                empty_streak = 0
            else:
                empty_streak += 1
                if empty_streak >= batch_docs * 5:
                    print(f"Warning: Too many empty rows on '{st['label']}', dropping stream.", flush=True)
                    exhausted = True
                    break

        if exhausted:
            print(f"Stream exhausted: {st['label']} ({st['docs']:,} docs read)", flush=True)
            streams.pop(idx)
            weights = np.delete(weights, idx)
            if len(weights) > 0:
                weights /= weights.sum()

        if not texts:
            continue

        # Parallel multi-threaded Rust BPE tokenization
        encodings = tokenizer.encode_batch(texts)
        st["docs"] += len(encodings)
        total_docs += len(encodings)

        # Route documents to train / val / test with <eos> boundary after every document
        train_batch_arrays: List[np.ndarray] = []
        for enc in encodings:
            ids = enc.ids
            if not ids:
                continue
            arr = np.empty(len(ids) + 1, dtype=np.uint16)
            arr[:-1] = ids
            arr[-1] = eos_id

            r = rng.random()
            if val_tokens < target_val and r < val_ratio:
                val_chunks.append(arr)
                val_tokens += len(arr)
            elif test_tokens < target_test and r < (val_ratio + test_ratio):
                test_chunks.append(arr)
                test_tokens += len(arr)
            else:
                train_batch_arrays.append(arr)

        if train_batch_arrays:
            batch_cat = np.concatenate(train_batch_arrays)
            prev_total = train_writer.total_tokens
            remaining = target_tokens - prev_total
            if len(batch_cat) > remaining:
                batch_cat = batch_cat[:remaining]
            train_writer.add(batch_cat)
            pbar.update(train_writer.total_tokens - prev_total)
            if train_writer.total_tokens >= next_log_tokens:
                dt = max(time.time() - t0, 1e-5)
                tps = train_writer.total_tokens / dt
                eta_min = (target_tokens - train_writer.total_tokens) / max(tps, 1.0) / 60.0
                print(
                    f"  [Progress] {train_writer.total_tokens:,}/{target_tokens:,} train tok "
                    f"({100.0 * train_writer.total_tokens / target_tokens:.1f}%) | "
                    f"Val: {val_tokens:,} | Test: {test_tokens:,} | "
                    f"Speed: {tps:,.0f} tok/s | ETA: {eta_min:.1f} min",
                    flush=True,
                )
                next_log_tokens += 25_000_000

    pbar.close()
    train_shards = train_writer.finalize()

    # Save val.bin and test.bin
    val_arr = np.concatenate(val_chunks) if val_chunks else np.array([], dtype=np.uint16)
    test_arr = np.concatenate(test_chunks) if test_chunks else np.array([], dtype=np.uint16)

    val_path = os.path.join(output_dir, "val.bin")
    test_path = os.path.join(output_dir, "test.bin")
    val_arr.tofile(val_path)
    test_arr.tofile(test_path)

    elapsed = time.time() - t0
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_token_id": eos_id,
        "train_tokens": int(train_writer.total_tokens),
        "val_tokens": int(len(val_arr)),
        "test_tokens": int(len(test_arr)),
        "train_shards": train_shards,
        "shard_size": shard_size,
        "total_docs": total_docs,
        "elapsed_sec": round(elapsed, 1),
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n==========================================")
    print(f"Done in {elapsed:.1f}s ({(train_writer.total_tokens + val_tokens + test_tokens) / max(elapsed, 1e-5):,.0f} tok/s)")
    print(f"Train: {train_writer.total_tokens:,} tokens across {len(train_shards)} shard(s)")
    print(f"Val:   {len(val_arr):,} tokens -> {val_path}")
    print(f"Test:  {len(test_arr):,} tokens -> {test_path}")
    print(f"Meta:  {meta_path}")
    print("==========================================")


def main():
    parser = argparse.ArgumentParser(description="Prepare sharded uint16 binary datasets with batched Rust tokenization.")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json.")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for shards and splits.")
    parser.add_argument("--target_tokens", type=int, default=1_000_000_000, help="Target number of training tokens.")
    parser.add_argument("--shard_size", type=int, default=100_000_000, help="Tokens per training shard (uint16).")
    parser.add_argument("--batch_docs", type=int, default=1024, help="Documents per Rust encode_batch call.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stream interleaving & split routing.")
    args = parser.parse_args()

    prepare_bin_data(
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        target_tokens=args.target_tokens,
        shard_size=args.shard_size,
        batch_docs=args.batch_docs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
