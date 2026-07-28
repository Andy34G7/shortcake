# Shortcake

`Shortcake` is a hybrid language model family (5M to 350M parameters) that interleaves **Mamba State Space Model (SSM)** layers and **Multi-Head Self-Attention** layers in a 3:1 ratio (emulating Nemotron-style hybrid backbones).

It features a custom Byte-Level BPE tokenizer with ChatML support, high-throughput binary memory-mapped data loading, memory-efficient PyTorch SSM selective scan with CUDA kernel support, gradient checkpointing for minimal VRAM footprint, selective weight decay, Supervised Fine-Tuning (SFT), terminal chat interface, and evaluation benchmark tools.

---

## Model Size Variants

Shortcake provides built-in model presets that scale from **~5M to ~350M parameters**:

| Variant Preset | Alias | Parameters | `d_model` | `n_layer` | `n_head` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`tiny`** | `5m` | **~5.2M** | 192 | 4 | 3 |
| **`small`** *(Default)* | `16m` / `20m` | **~16.4M** | 384 | 8 | 6 |
| **`base`** | `50m` | **~48.5M** | 512 | 12 | 8 |
| **`medium`** | `110m` | **~108.2M** | 768 | 12 | 12 |
| **`large`** | `350m` | **~345.8M** | 1024 | 24 | 16 |

---

## Model Architecture & Optimizations

- **Layer Ratio**: 3 Mamba layers per 1 Self-Attention layer.
- **Selective Scan Engine**: Memory-efficient native PyTorch selective scan with automatic fallback to fused `mamba-ssm` CUDA kernels when installed.
- **Gradient Checkpointing**: Discards intermediate sequence step autograd graph nodes during training, keeping VRAM usage under **< 1.5 GB** at batch sizes of 16, 32, or higher.
- **Preserved SSM Step-Size Initialization**: Preserves `dt_proj.bias` initialization (`_reset_dt_bias()`) to guarantee calibrated Mamba state step sizes ($\Delta \approx 0.001..0.1$) across deep sequences.
- **Selective Weight Decay**: Configures AdamW to apply weight decay (0.01..0.1) exclusively to 2D matrix weights, protecting 1D RMSNorm gain weights, biases, $A\_log$, $D$, and $dt\_proj$ bias.
- **Attention Engine**: PyTorch `scaled_dot_product_attention` (SDPA / FlashAttention-2).
- **Tokenizer**: Custom Byte-Level BPE (~16,384 vocabulary) with ChatML special tokens (`<|im_start|>`, `<|im_end|>`).

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
