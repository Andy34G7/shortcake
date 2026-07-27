# Shortcake

`Shortcake` is an ultra-compact hybrid language model (16,406,400 parameters) that interleaves **Mamba State Space Model (SSM)** layers and **Multi-Head Self-Attention** layers in a 3:1 ratio (emulating Nemotron-style hybrid backbones).

It features a custom Byte-Level BPE tokenizer, high-throughput binary memory-mapped data loading, memory-efficient PyTorch SSM selective scan, gradient checkpointing for minimal VRAM footprint, selective weight decay parameter classification, and a complete train/validate/test evaluation harness.

---

## Model Architecture & Performance Optimizations

- **Target Parameters**: ~20M (`d_model` = 384, 8 layers, 6 heads, `d_state` = 16).
- **Layer Ratio**: 3 Mamba layers per 1 Self-Attention layer.
- **Selective Scan Engine**: Memory-efficient native PyTorch selective scan with automatic fallback to fused `mamba-ssm` CUDA kernels when installed.
- **Gradient Checkpointing**: Discards intermediate sequence step autograd graph nodes during training, keeping VRAM usage under **< 1.5 GB** (enabling training at batch sizes of 16, 32, or higher without CUDA OOM).
- **Preserved SSM Step-Size Initialization**: Preserves `dt_proj.bias` initialization (`_reset_dt_bias()`) to guarantee calibrated Mamba state step sizes ($\Delta \approx 0.001..0.1$) across deep sequences.
- **Selective Weight Decay**: Configures AdamW to apply weight decay (0.1) exclusively to 2D matrix weights, protecting 1D RMSNorm gain weights, biases, $A\_log$, $D$, and $dt\_proj$ bias.
- **Attention Engine**: PyTorch `scaled_dot_product_attention` (SDPA / FlashAttention-2).
- **Tokenizer**: Custom Byte-Level BPE (~16,384 vocabulary) to minimize parameter budget spent on embeddings.

---

## Repository Structure

```
smol-transformer/
├── config.py            # Model hyperparameter configuration
├── mamba.py             # Memory-efficient PyTorch Mamba SSM block & CUDA kernel interface
├── model.py             # Shortcake architecture (Mamba + Attention)
├── train_tokenizer.py   # Byte-Level BPE tokenizer trainer
├── prepare_data.py      # Tokenizes datasets to uint16 memmap binary splits
├── train.py             # Train/Val/Test harness with checkpoint saving & resume
├── train_modal.py       # Modal GPU serverless training runner
├── generate.py          # Code completion sampling script
├── test_suite.py        # Complete automated test suite
├── pyproject.toml       # Project configuration & dependencies
├── requirements.txt     # Dependency list for pip / DigitalOcean
└── README.md            # Usage and setup guide
```

---

## Training Datasets

The tokenizer and data preparation pipeline automatically stream across code datasets:

1. **SmolLM-Corpus** (`HuggingFaceTB/SmolLM-Corpus` - Python)
2. **CodeParrot** (`codeparrot/codeparrot-clean`)

---

## Complete Workflow on DigitalOcean (or Local Server)

### 1. Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using standard pip
pip install -r requirements.txt
```

### 2. Train Tokenizer
Stream across the datasets automatically to train your 16,384 vocabulary Byte-Level BPE tokenizer:
```bash
uv run python train_tokenizer.py
```

### 3. Stream & Tokenize Datasets into Binary Splits
Tokenize data into high-throughput uint16 `train.bin`, `val.bin`, and `test.bin` splits:
```bash
uv run python prepare_data.py
```

### 4. Run Pre-Training Harness
```bash
uv run python train.py --max_steps 10000 --eval_interval 100 --save_interval 500 --batch_size 16 --lr 6e-4
```

To resume from the latest saved checkpoint:
```bash
uv run python train.py --resume checkpoints/latest.pt
```

---

## Modal Cloud Training (Optional GPU Acceleration)

To run GPU pre-training on Modal:

```bash
# Prepare dataset on Modal Volume
uv run modal run train_modal.py --action prepare

# Run GPU pre-training (A10G GPU)
uv run modal run train_modal.py --action train --batch-size 16

# Resume from existing Modal Volume checkpoint
uv run modal run train_modal.py --action train --batch-size 16 --resume

# Download best checkpoint to local ./checkpoints/ directory
uv run modal run train_modal.py --action download
```

---

## Code Completion Sampling

Sample code completions from a trained model checkpoint:
```bash
uv run python generate.py --checkpoint checkpoints/best_model.pt --prompt "def binary_search("
```

---

## Automated Test Suite

Verify model architecture, tokenizer, training step loss reduction, and SPLADE head:
```bash
uv run python test_suite.py
```
