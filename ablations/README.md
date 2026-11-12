# Architecture ablations

A small, CPU-friendly harness for comparing Shortcake design choices (RoPE, init, layer
layout) on a tiny model. It is meant for quick directional checks, not for final numbers.

## Running

From the repo root:

```bash
# 1. Build the corpus and tokenizer (~10 s to a minute): Python files from the local
#    standard library, ~24M characters, 4,096-token BPE via train_tokenizer.py.
python ablations/prep.py

# 2. Train one or more variants (defaults reproduce the table below).
python ablations/abl.py --variant baseline rope+init

# Parameter counts only.
python ablations/abl.py --variant all --count
```

Everything generated (tokenizer, `train.bin`/`val.bin`, per-run JSON in `results/`) goes
to `ablations/out/`, which is git-ignored. `prep.py` stubs the `datasets` package if it is not
installed, since `train_tokenizer.py` imports it but the function used here does not need it.

## Variants

Layout strings, one character per layer:

| char | layer |
|------|-------|
| `M`  | Mamba + SwiGLU MLP |
| `m`  | Mamba only |
| `A`  | attention + SwiGLU MLP |
| `a`  | attention only |

Each variant also sets `rope` (rotary embeddings on attention q/k) and `init`:
`orig` is the old repo init (every Linear, including Mamba's `dt_proj`, re-initialised to
N(0, 0.02)); `fixed` keeps `MambaBlock`'s `dt_proj` init and scales the residual output
projections (`out_proj`, `mlp.w2`) by 1/sqrt(2 * n_layers).

- `baseline` = `Variant("MMMA", rope=False, init="orig")`: the repo model before RoPE and the
  init fix (same parameters and initial weights as the old `Shortcake` for a given seed).
- `repo` = the current `model.Shortcake`, to check it against the ablation variants.
- `rope`, `init`, `rope+init`: MMMA with each change; the other layouts all use RoPE + init fix.

See `VARIANTS` in `abl.py` for the full list.

## Results

Setup: d_model 192, 4 layers (layouts set their own depth), 3 heads, vocab 4096, sequence
length 256, batch 8, 1,200 steps (100 warmup, cosine to lr/10), peak lr 6e-4, weight decay 0.1,
one seed (0), CPU. Validation loss is the mean over 20 fixed batches of held-out files.

| variant | layout | RoPE | init | non-embedding params | val loss @1200 |
|---------|--------|------|------|---------------------:|---------------:|
| baseline | MMMA | no | orig | 2.08M | 3.483 |
| rope | MMMA | yes | orig | 2.08M | 3.460 |
| init | MMMA | no | fixed | 2.08M | 3.444 |
| rope+init | MMMA | yes | fixed | 2.08M | **3.429** |
| MMAM | MMAM | yes | fixed | 2.08M | 3.452 |
| MAMA | MAMA | yes | fixed | 1.98M | 3.471 |
| mmmAmmA | mmmAmmA | yes | fixed | 2.14M | 3.455 |
| mAmAmA | mAmAmA | yes | fixed | 2.08M | 3.492 |

**These are single-seed, short runs of a tiny model, so treat them as directional only.**
Differences of a few hundredths may be within seed-to-seed noise, and nothing here has been
checked at a larger scale or for longer training. With that caveat: RoPE and the init fix each
helped and stacked, and the default MMMA layout did best among the layouts tried.

The corpus depends on the local Python version and on the order `os.walk` lists files, so
another machine will build a slightly different dataset and the exact numbers will move.
