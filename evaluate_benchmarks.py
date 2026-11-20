import os
import time
import math
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoModel

from config import ModelConfig
from model import Shortcake
from train import BinaryMemmapDataLoader, evaluate


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def evaluate_checkpoint_perplexity(
    checkpoint_path: str,
    data_dir: str = "data",
    batch_size: int = 4,
    eval_iters: int = 30,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Shortcake, Optional[Tuple[float, float]]]:
    """Load a trained Shortcake checkpoint and evaluate real test set loss & perplexity."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", ModelConfig())
    model = Shortcake(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint '{checkpoint_path}' ({model.get_num_params():,} non-embedding params).")

    test_bin = os.path.join(data_dir, "test.bin")
    if not os.path.exists(test_bin):
        print(f"Notice: '{test_bin}' not found; skipping test perplexity evaluation.")
        return model, None

    use_bf16 = (device == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32)
    test_loader = BinaryMemmapDataLoader(test_bin, config.max_seq_len, batch_size, device)
    test_loss, test_ppl = evaluate(model, test_loader, eval_iters=eval_iters, device=device, amp_dtype=amp_dtype)
    print(f">>> Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.2f}")
    return model, (test_loss, test_ppl)


def benchmark_decode_cache_speedup(
    model: Shortcake,
    prompt_len: int = 128,
    gen_tokens: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """Measure real generation throughput (tokens/sec) with vs without SSM+KV inference cache."""
    model.eval()
    prompt = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)

    with torch.no_grad():
        # Uncached baseline
        ids = prompt.clone()
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(gen_tokens):
            logits = model(ids)["logits"][:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        _sync(device)
        uncached_tps = gen_tokens / max(time.perf_counter() - t0, 1e-6)

        # Cached (SSM state + KV cache)
        cache = model.init_cache()
        _sync(device)
        t0 = time.perf_counter()
        logits = model(prompt, cache=cache)["logits"][:, -1, :]
        for _ in range(gen_tokens):
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            logits = model(nxt, cache=cache)["logits"][:, -1, :]
        _sync(device)
        cached_tps = gen_tokens / max(time.perf_counter() - t0, 1e-6)

    print(
        f"Decode Throughput (prompt={prompt_len}, gen={gen_tokens}): "
        f"Uncached={uncached_tps:.1f} tok/s | Cached={cached_tps:.1f} tok/s "
        f"({cached_tps / max(uncached_tps, 1e-6):.1f}x speedup)"
    )
    return {"Uncached": uncached_tps, "Cached (SSM+KV)": cached_tps}


def benchmark_throughput_vs_seq_len(
    model: Optional[Shortcake] = None,
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, List[float]]:
    """Benchmark forward throughput (tokens/sec) across sequence lengths with proper CUDA sync."""
    print("--- Running Forward Throughput Benchmark across Sequence Lengths ---")

    if model is None:
        config = ModelConfig(vocab_size=16384, d_model=384, n_layer=8, n_head=6)
        model = Shortcake(config).to(device)
    model.eval()
    config = model.config

    results: Dict[str, List[float]] = {"Shortcake": []}

    bert_model = None
    try:
        bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
        results["BERT-base (~110M)"] = []
    except Exception as e:
        print(f"Notice: Skipping BERT-base baseline ({e})")

    smollm_model = None
    try:
        smollm_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").to(device).eval()
        results["SmolLM2-135M"] = []
    except Exception as e:
        print(f"Notice: Skipping SmolLM2-135M baseline ({e})")

    iters = 5
    with torch.no_grad():
        for L in seq_lengths:
            print(f"Testing sequence length L = {L}...")

            dummy_input = torch.randint(0, config.vocab_size, (batch_size, L), device=device)
            for _ in range(2):
                _ = model(dummy_input)
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(dummy_input)
            _sync(device)
            t1 = time.perf_counter()
            results["Shortcake"].append((batch_size * L * iters) / max(t1 - t0, 1e-6))

            if bert_model is not None:
                bert_L = min(L, 512)
                dummy_bert = torch.randint(0, 30000, (batch_size, bert_L), device=device)
                _sync(device)
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = bert_model(dummy_bert)
                _sync(device)
                t1 = time.perf_counter()
                results["BERT-base (~110M)"].append((batch_size * bert_L * iters) / max(t1 - t0, 1e-6))

            if smollm_model is not None:
                dummy_smol = torch.randint(0, 49152, (batch_size, L), device=device)
                _sync(device)
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = smollm_model(dummy_smol)
                _sync(device)
                t1 = time.perf_counter()
                results["SmolLM2-135M"].append((batch_size * L * iters) / max(t1 - t0, 1e-6))

    return results


def plot_benchmark_results(
    seq_lengths: List[int],
    throughput_results: Dict[str, List[float]],
    decode_results: Dict[str, float],
    output_png: str = "benchmark_comparison.png",
):
    """Plot measured forward throughput vs sequence length and cached decode speedup."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    styles = {
        "Shortcake": ("#FF4B4B", "o-", 2.5),
        "BERT-base (~110M)": ("#4B70FF", "s--", 1.8),
        "SmolLM2-135M": ("#2ECC71", "^-.", 1.8),
    }

    for model_name, throughput in throughput_results.items():
        color, linestyle, lw = styles.get(model_name, ("#333333", "-", 1.5))
        ax1.plot(seq_lengths, throughput, linestyle, label=model_name, color=color, linewidth=lw, markersize=7)

    ax1.set_title("Prefill Throughput vs Context Length", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sequence Length (Tokens)", fontsize=10)
    ax1.set_ylabel("Throughput (Tokens / sec)", fontsize=10)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(seq_lengths)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=9)

    labels = list(decode_results.keys())
    values = list(decode_results.values())
    colors = ["#95A5A6", "#FF4B4B"]
    bars = ax2.bar(labels, values, color=colors[: len(labels)], width=0.45, edgecolor="black", linewidth=1.0)
    ax2.set_title("Shortcake Autoregressive Decode Throughput", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Tokens / sec", fontsize=10)
    ax2.grid(axis="y", ls="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight")
    print(f"\n Benchmark graphs successfully saved to '{output_png}'!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Shortcake test perplexity and throughput benchmarks.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint.pt for test perplexity eval.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing test.bin.")
    parser.add_argument("--eval_iters", type=int, default=30, help="Number of test batches for perplexity evaluation.")
    parser.add_argument("--skip_plot", action="store_true", help="Skip saving matplotlib PNG chart.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Shortcake Benchmark & Evaluation ({device}) ===")

    model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        model, _ = evaluate_checkpoint_perplexity(
            args.checkpoint, data_dir=args.data_dir, eval_iters=args.eval_iters, device=device
        )
    else:
        config = ModelConfig()
        model = Shortcake(config).to(device).eval()

    decode_results = benchmark_decode_cache_speedup(model, device=device)
    seq_lengths = [512, 1024, 2048]
    throughput_results = benchmark_throughput_vs_seq_len(model=model, seq_lengths=seq_lengths, device=device)

    if not args.skip_plot:
        plot_benchmark_results(seq_lengths, throughput_results, decode_results)


if __name__ == "__main__":
    main()
