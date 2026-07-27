# Shortcake

`Shortcake` is an ultra-compact hybrid language model (16,406,400 parameters) that interleaves **Mamba State Space Model (SSM)** layers and **Multi-Head Self-Attention** layers in a 3:1 ratio (emulates Nemotron).

It features a custom Byte-Level BPE tokenizer, high-throughput binary memory-mapped dataloading, a full train/validate/test evaluation harness with checkpointing.

---

## Model Architecture

- **Target Parameters**: ~20M (`d_model` = 384, 8 layers, 6 heads, `d_state` = 16).
- **Layer Ratio**: 3 Mamba layers per 1 Self-Attention layer.
- **Mamba Engine**: Pure PyTorch native selective scan with automatic fallback to compiled `mamba-ssm` CUDA kernels if installed.
- **Attention Engine**: PyTorch `scaled_dot_product_attention` (SDPA / FlashAttention-2).
- **Tokenizer**: Custom Byte-Level BPE (~16,384 vocabulary) to minimize parameter budget spent on embeddings.

---

## Repository Structure

```
smol-hybrid-transformer/
├── config.py            # Model hyperparameter configuration
├── mamba.py             # Pure PyTorch Mamba SSM block & CUDA fallback
├── model.py             # Shortcake architecture (Mamba + Attention)
├── train_tokenizer.py   # Byte-Level BPE tokenizer trainer
├── prepare_data.py      # Tokenizes datasets to uint16 memmap binary splits
├── train.py             # Train/Val/Test harness with checkpoint saving & resume
├── generate.py          # Code completion sampling script
├── test_suite.py        # Complete automated test suite
├── requirements.txt     # Dependency list
└── README.md            # Usage and setup guide
```

---

## Hardcoded Training Datasets

The tokenizer and data preparation pipeline automatically stream across 3 core code datasets:

1. **SmolLM-Corpus** (`HuggingFaceTB/SmolLM-Corpus` - Python)
2. **CodeParrot** (`codeparrot/codeparrot-clean`)
3. **The Stack** (`bigcode/the-stack` - Python)

---

## Complete Workflow on DigitalOcean

### 1. Train Tokenizer
Stream across the 3 datasets automatically to train your 16,384 vocabulary Byte-Level BPE tokenizer:
```bash
uv run python train_tokenizer.py
```

### 2. Stream & Tokenize Datasets into Binary Splits
Stream documents across all 3 datasets and tokenize into high-throughput uint16 `train.bin`, `val.bin`, and `test.bin` files:
```bash
uv run python prepare_data.py
```

### 3. Run Pre-Training Harness
```bash
uv run python train.py --max_steps 5000 --eval_interval 250 --save_interval 500 --batch_size 8 --lr 6e-4
```


To resume from the latest saved checkpoint:
```bash
python train.py --resume checkpoints/latest.pt
```

### 5. Generate Code Completions

Sample code generation from your best trained checkpoint:
```bash
python generate.py --checkpoint checkpoints/best_model.pt --prompt "def binary_search("
```

### 6. Run Test Suite

Verify architecture, tokenizer, training step loss reduction, and SPLADE head:
```bash
python test_suite.py
```

