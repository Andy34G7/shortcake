"""Small-scale architecture ablations for Shortcake.

Trains a tiny model on the stdlib corpus from prep.py and reports validation loss.
Each variant is a layer layout string plus two switches:

    M = Mamba + MLP block      m = Mamba block only
    A = attention + MLP block  a = attention block only

    rope  rotary position embeddings on attention q/k
    init  "orig"  = old repo init: every Linear (incl. Mamba dt_proj) ~ N(0, 0.02)
          "fixed" = keep MambaBlock's dt_proj init, scale residual output projections
                    (out_proj, mlp.w2) by 1/sqrt(2 * n_layers)

"baseline" rebuilds the old repo model (MMMA, no RoPE, orig init); "repo" instantiates the
current `model.Shortcake` so the two can be compared.

Run from the repo root (after `python ablations/prep.py`):

    python ablations/abl.py --variant baseline rope+init --no-checkpoint
    python ablations/abl.py --variant all --count
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ABLATIONS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ABLATIONS_DIR.parent
DEFAULT_OUT_DIR = ABLATIONS_DIR / "out"
sys.path.insert(0, str(REPO_ROOT))

from config import ModelConfig  # noqa: E402
from mamba import MambaBlock  # noqa: E402
from model import RMSNorm, Shortcake, SwiGLUMLP  # noqa: E402
from train import configure_optimizers, get_lr  # noqa: E402


# --------------------------------------------------------------------------- model

class AttentionLayer(nn.Module):
    """Pre-norm causal self-attention, optional RoPE, optional SwiGLU MLP."""

    def __init__(self, cfg: ModelConfig, rope: bool, mlp: bool):
        super().__init__()
        self.n_head, self.head_dim, self.rope = cfg.n_head, cfg.head_dim, rope
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLUMLP(cfg) if mlp else None
        self.mlp_norm = RMSNorm(cfg.d_model) if mlp else None
        if rope:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            angles = torch.outer(torch.arange(cfg.max_seq_len).float(), inv_freq)  # (L, head_dim/2)
            self.register_buffer("cos", angles.cos(), persistent=False)
            self.register_buffer("sin", angles.sin(), persistent=False)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate interleaved (even, odd) channel pairs of x: (B, H, L, head_dim)."""
        seq_len = x.shape[2]
        cos, sin = self.cos[:seq_len], self.sin[:seq_len]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        q, k, v = self.qkv_proj(self.norm(x)).chunk(3, dim=-1)
        q, k, v = (t.view(b, l, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if self.rope:
            q, k = self.apply_rope(q), self.apply_rope(k)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out_proj(attn.transpose(1, 2).reshape(b, l, d))
        if self.mlp is not None:
            x = x + self.mlp(self.mlp_norm(x))
        return x


class MambaLayer(nn.Module):
    """Pre-norm MambaBlock with residual, optional SwiGLU MLP."""

    def __init__(self, cfg: ModelConfig, mlp: bool):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mamba = MambaBlock(cfg)
        self.mlp = SwiGLUMLP(cfg) if mlp else None
        self.mlp_norm = RMSNorm(cfg.d_model) if mlp else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba(self.norm(x))
        if self.mlp is not None:
            x = x + self.mlp(self.mlp_norm(x))
        return x


class VariantModel(nn.Module):
    """Tied-embedding LM built from a layout string (see module docstring)."""

    def __init__(self, cfg: ModelConfig, layout: str, rope: bool, init: str):
        super().__init__()
        assert init in ("orig", "fixed"), init
        assert set(layout) <= set("MmAa"), layout
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            MambaLayer(cfg, mlp=(ch == "M")) if ch in "Mm" else AttentionLayer(cfg, rope, mlp=(ch == "A"))
            for ch in layout
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        # Same order of RNG draws as the old Shortcake.__init__, so "baseline" reproduces it.
        dt_weights = {id(m): m.dt_proj.weight.detach().clone() for m in self.modules() if isinstance(m, MambaBlock)}
        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, MambaBlock):
                m._reset_dt_bias()
                if init == "fixed":
                    with torch.no_grad():
                        m.dt_proj.weight.copy_(dt_weights[id(m)])
        if init == "fixed":
            residual_std = 0.02 / math.sqrt(2 * len(layout))
            for name, p in self.named_parameters():
                if name.endswith("out_proj.weight") or name.endswith("mlp.w2.weight"):
                    nn.init.normal_(p, 0.0, residual_std)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor) -> dict:
        h = self.tok_embeddings(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.norm_f(h))
        return {"loss": F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))}


@dataclass(frozen=True)
class Variant:
    layout: str = ""        # empty => use the repo's current Shortcake
    rope: bool = True
    init: str = "fixed"

    def build(self, cfg: ModelConfig) -> nn.Module:
        if not self.layout:
            return Shortcake(cfg)
        return VariantModel(cfg, self.layout, self.rope, self.init)


VARIANTS = {
    "baseline": Variant("MMMA", rope=False, init="orig"),  # the repo model before RoPE + init fix
    "repo": Variant(),                                     # current model.Shortcake
    "rope": Variant("MMMA", rope=True, init="orig"),
    "init": Variant("MMMA", rope=False, init="fixed"),
    "rope+init": Variant("MMMA", rope=True, init="fixed"),
    # layouts, all with RoPE + init fix
    "MAMA": Variant("MAMA"),
    "MMAM": Variant("MMAM"),
    "AMMA": Variant("AMMA"),
    "mmmAmmA": Variant("mmmAmmA"),
    "mmAmmA": Variant("mmAmmA"),
    "mamaMA": Variant("mamaMA"),
    "mAmAmA": Variant("mAmAmA"),
    "MAMAA": Variant("MAMAA"),
}


# --------------------------------------------------------------------------- training

def disable_scan_checkpointing():
    """Make torch.utils.checkpoint.checkpoint a plain call.

    MambaBlock wraps the pure-PyTorch scan in checkpoint() during training to save memory;
    the tiny ablation models do not need that, and skipping the recompute is faster.
    This patches torch globally, which is fine for this standalone script.
    """
    import torch.utils.checkpoint as checkpoint_module

    checkpoint_module.checkpoint = lambda fn, *args, use_reentrant=False, **kwargs: fn(*args, **kwargs)


def param_counts(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())  # tied lm_head counted once
    return total, total - model.tok_embeddings.weight.numel()


def train_variant(name: str, args) -> dict:
    torch.manual_seed(args.seed)
    model = VARIANTS[name].build(make_config(args))
    total, non_emb = param_counts(model)
    print(f"{name}: {total / 1e6:.3f}M params ({non_emb / 1e6:.3f}M non-embedding)", flush=True)

    train_data = np.memmap(args.out_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(args.out_dir / "val.bin", dtype=np.uint16, mode="r")
    T = args.seq_len

    def get_batch(data, starts):
        x = np.stack([data[i : i + T] for i in starts]).astype(np.int64)
        y = np.stack([data[i + 1 : i + 1 + T] for i in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    # Fixed validation batches (same for every variant and seed).
    val_starts = np.random.default_rng(7).integers(0, len(val_data) - T - 1, size=(args.val_batches, args.batch_size))
    train_rng = np.random.default_rng(1000 + args.seed)

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        losses = [model(*get_batch(val_data, starts))["loss"].item() for starts in val_starts]
        model.train()
        return float(np.mean(losses))

    optimizer = configure_optimizers(model, args.lr, args.weight_decay)
    result = {"variant": name, "config": VARIANTS[name].__dict__, "seed": args.seed, "params": total,
              "non_embedding_params": non_emb, "args": {k: str(v) for k, v in vars(args).items()}, "val": {}}
    t0 = time.time()
    ema = None
    for step in range(args.steps + 1):
        lr = get_lr(step, args.warmup, args.steps, args.lr, args.lr / 10)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, train_rng.integers(0, len(train_data) - T - 1, size=args.batch_size))
        loss = model(x, targets=y)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step > 0 and (step % args.eval_every == 0 or step == args.steps):
            val = evaluate()
            result["val"][step] = val
            print(f"{name} seed {args.seed} step {step} train_ema {ema:.4f} val {val:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    result["time_s"] = time.time() - t0

    results_dir = args.out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{name}_s{args.seed}_d{args.d_model}.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(f"wrote {out_path}")
    return result


def make_config(args) -> ModelConfig:
    return ModelConfig(vocab_size=args.vocab_size, d_model=args.d_model, n_layer=args.n_layer,
                       n_head=args.n_head, max_seq_len=args.seq_len)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", nargs="+", required=True, help=f"one or more of: all, {', '.join(VARIANTS)}")
    ap.add_argument("--count", action="store_true", help="only print parameter counts")
    ap.add_argument("--no-checkpoint", action="store_true", help="disable gradient checkpointing of the Mamba scan (faster)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-head", type=int, default=3)
    ap.add_argument("--n-layer", type=int, default=4, help="used by the 'repo' variant; layouts set their own depth")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    names = list(VARIANTS) if args.variant == ["all"] else args.variant
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        ap.error(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")

    if args.count:
        cfg = make_config(args)
        for name in names:
            torch.manual_seed(args.seed)
            total, non_emb = param_counts(VARIANTS[name].build(cfg))
            print(f"{name:12s} {VARIANTS[name].layout or 'Shortcake':10s} total={total / 1e6:.3f}M  non-emb={non_emb / 1e6:.3f}M")
        return

    if not (args.out_dir / "train.bin").exists():
        sys.exit(f"{args.out_dir / 'train.bin'} not found; run `python ablations/prep.py` first")
    torch.set_num_threads(args.threads)
    if args.no_checkpoint:
        disable_scan_checkpointing()
    for name in names:
        train_variant(name, args)


if __name__ == "__main__":
    main()
