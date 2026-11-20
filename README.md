# Shortcake

`Shortcake` is a hybrid language model family (5M to 350M parameters) that interleaves **Mamba State Space Model (SSM)** layers and **Multi-Head Self-Attention** layers in a 3:1 ratio (emulating Nemotron-style hybrid backbones).

It features a custom Byte-Level BPE tokenizer with ChatML support, high-throughput binary memory-mapped data loading, memory-efficient PyTorch SSM selective scan with CUDA kernel support, gradient checkpointing for minimal VRAM footprint, selective weight decay, Supervised Fine-Tuning (SFT), terminal chat interface, and evaluation benchmark tools.

---

## Model Size Variants

Shortcake provides built-in model presets that scale from **~5M to ~350M parameters**:

| Variant Preset | Alias | Parameters | `d_model` | `n_layer` | `n_head` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`tiny`** | `5m` | **~2.1M** | 192 | 4 | 3 |
| **`small`** *(Default)* | `16m` / `20m` | **~16.4M** | 384 | 8 | 6 |
| **`base`** | `50m` | **~44.3M** | 512 | 12 | 8 |
| **`medium`** | `110m` | **~97.6M** | 768 | 12 | 12 |
| **`large`** | `350m` | **~348.1M** | 1024 | 24 | 16 |

---

## Model Architecture & Optimizations

- **Layer Ratio**: 3 Mamba layers per 1 Self-Attention layer.
- **Selective Scan Engine**: 3-tier backend — fused `mamba-ssm` + `causal-conv1d` CUDA kernels when installed, built-in **Triton fused CUDA selective scan** (~183k tok/s on A100), and a chunked parallel associative scan fallback in `float32`.
- **SSM + KV Inference Cache**: `model.init_cache()` caches per-layer Mamba `(conv_state, ssm_state)` for $O(1)$ recurrent decoding alongside Attention `(K, V)` caches.
- **Attention Engine**: PyTorch `scaled_dot_product_attention` (SDPA / FlashAttention-2) with **RoPE** (Rotary Position Embeddings) and **QK-Norm** (`RMSNorm`).
- **Calibrated Initialization**: Multi-scale S4D `A_log` initialization (`[1..d_state]` per channel), preserved `dt_proj` weight & bias initialization, and `1 / sqrt(2 * n_layer)` residual projection scaling.
- **Selective Weight Decay & Precision**: AdamW applies weight decay only to 2D linear weights; automatic `bfloat16` autocast with `float32`-safe `RMSNorm` and SSM state accumulation.
- **Sharded Data Pipeline**: Batched Rust BPE tokenization (`encode_batch`) with weighted-interleaved streaming into 100M-token `uint16` shards and `<eos>` document boundaries.
- **Tokenizer**: Custom Byte-Level BPE (16,384 vocabulary) with ChatML special tokens (`<|im_start|>`, `<|im_end|>`).

---

## v0.2.0 Benchmark Results (2B-Token Corpus, NVIDIA A100-40GB)

| Preset | Non-Emb / Total Params | Pre-train Tokens | Test Loss | **Test PPL** | **SFT Val Loss** (`smoltalk`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`tiny`** | 2.1M / ~5.2M | 262M | `2.9682` | **`19.46`** | **`2.6288`** |
| **`small`** | 16.4M / ~22.7M | 655M | `2.3302` | **`10.28`** | **`2.0348`** |
| **`base`** | 44.4M / ~52.8M | 1.31B | `2.0979` | **`8.15`** | **`1.8513`** |

---

## Repository Structure

```
smol-transformer/
├── config.py            # Model config & size variant presets (tiny, small, base, medium, large)
├── mamba.py             # Memory-efficient PyTorch Mamba SSM block & CUDA kernel interface
├── model.py             # Shortcake architecture (Mamba + Attention)
├── train_tokenizer.py   # Byte-Level BPE tokenizer trainer (with ChatML special tokens)
├── prepare_data.py      # Tokenizes datasets to uint16 memmap binary splits
├── train.py             # Pre-training harness with checkpoint saving & resume
├── train_sft.py         # Supervised Fine-Tuning (SFT) on ChatML instruction datasets
├── chat.py              # Interactive terminal chat CLI
├── generate.py          # Code/text sampling script (with Top-K, Top-P, Repetition Penalty)
├── evaluate_benchmarks.py # Test loss/perplexity and Python execution pass@k benchmarks
├── publish_to_hf.py     # Script to publish model card and weights to Hugging Face Hub
├── test_suite.py        # Automated unit test suite
├── pyproject.toml       # Project configuration & dependencies
├── requirements.txt     # Dependency list
└── README.md            # Usage and setup guide
```

---

## Complete Setup & Training Workflow

### 1. Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using standard pip
pip install -r requirements.txt
```

### 2. Train Tokenizer
Stream across datasets automatically to train your 16,384 vocabulary Byte-Level BPE tokenizer with ChatML tokens:
```bash
uv run python train_tokenizer.py
```

### 3. Stream & Tokenize Datasets into Binary Splits
Tokenize data into high-throughput uint16 `train.bin`, `val.bin`, and `test.bin` splits:
```bash
uv run python prepare_data.py
```

### 4. Run Pre-Training
Train any variant size (e.g. `small` / 16M, `base` / 50M, `medium` / 110M):
```bash
# Train default small (16M) variant
uv run python train.py --preset small --max_steps 10000 --batch_size 16 --lr 6e-4

# Train base (50M) variant
uv run python train.py --preset base --max_steps 10000 --batch_size 16 --lr 4e-4
```

To resume training from the latest checkpoint:
```bash
uv run python train.py --resume checkpoints/latest.pt
```

---

## Supervised Fine-Tuning (SFT) & Chat CLI

Fine-tune Shortcake on conversational instruction datasets (`HuggingFaceTB/smoltalk`) using ChatML format:

```bash
# Run Instruction Fine-Tuning
uv run python train_sft.py --base_checkpoint checkpoints/best_model.pt --max_steps 2000 --batch_size 16 --lr 2e-5

# Start Interactive Terminal Chat
uv run python chat.py --checkpoint checkpoints_sft/best_sft_model.pt
```

---

## Benchmark Evaluation

Evaluate test set perplexity and Python code execution accuracy:
```bash
uv run python evaluate_benchmarks.py --checkpoint checkpoints/best_model.pt
```

---

## Text & Code Generation

Sample completions from a trained model checkpoint:
```bash
uv run python generate.py --checkpoint checkpoints/best_model.pt --prompt "def binary_search(" --repetition_penalty 1.2
```

---

## Automated Test Suite

Verify model architecture, size presets, tokenizer, loss reduction, and SPLADE head:
```bash
uv run python test_suite.py
```
