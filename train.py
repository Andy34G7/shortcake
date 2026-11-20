import os
import glob
import time
import math
import argparse
import traceback
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List

from config import ModelConfig
from model import Shortcake, RMSNorm
from mamba import HAS_MAMBA_CUDA, HAS_CAUSAL_CONV1D, HAS_TRITON


class BinaryMemmapDataLoader:
    """High-throughput memory-mapped binary dataset iterator supporting single files and multi-shard splits."""

    def __init__(self, filename: str, block_size: int, batch_size: int, device: str):
        shard_paths: List[str] = []
        if os.path.isfile(filename):
            shard_paths = [filename]
        else:
            # Auto-discover sharded splits: e.g. 'data/train.bin' -> 'data/train_*.bin'
            if "*" in filename:
                shard_paths = sorted(glob.glob(filename))
            elif filename.endswith(".bin"):
                pattern = filename[:-4] + "_*.bin"
                shard_paths = sorted(glob.glob(pattern))

        if not shard_paths:
            raise FileNotFoundError(
                f"Dataset file or shards for '{filename}' do not exist. Run prepare_data.py first!"
            )

        self.shards: List[np.memmap] = []
        self.max_indices: List[int] = []
        for path in shard_paths:
            mm = np.memmap(path, dtype=np.uint16, mode="r")
            max_idx = len(mm) - block_size - 1
            if max_idx > 0:
                self.shards.append(mm)
                self.max_indices.append(max_idx)

        if not self.shards:
            raise ValueError(
                f"Dataset in '{filename}' has insufficient tokens for block size {block_size}."
            )

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.total_tokens = sum(len(s) for s in self.shards)
        self.tokens_per_batch = block_size * batch_size

        weights = np.array(self.max_indices, dtype=np.float64)
        self.shard_probs = weights / weights.sum()

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch a random batch of input (x) and target (y) sequences across all shards."""
        if len(self.shards) == 1:
            data = self.shards[0]
            ix = np.random.randint(0, self.max_indices[0], size=self.batch_size)
            x_np = np.stack([data[i : i + self.block_size] for i in ix]).astype(np.int64)
            y_np = np.stack([data[i + 1 : i + 1 + self.block_size] for i in ix]).astype(np.int64)
        else:
            shard_ids = np.random.choice(len(self.shards), size=self.batch_size, p=self.shard_probs)
            x_list = []
            y_list = []
            for sid in shard_ids:
                data = self.shards[sid]
                i = int(np.random.randint(0, self.max_indices[sid]))
                x_list.append(data[i : i + self.block_size])
                y_list.append(data[i + 1 : i + 1 + self.block_size])
            x_np = np.stack(x_list).astype(np.int64)
            y_np = np.stack(y_list).astype(np.int64)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y


def get_lr(it: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if warmup_steps > 0 and it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it >= max_steps or max_steps <= warmup_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(
    model: Shortcake,
    dataloader: BinaryMemmapDataLoader,
    eval_iters: int = 10,
    device: str = "cpu",
    amp_dtype: torch.dtype = torch.float32,
) -> Tuple[float, float]:
    """Evaluate model loss and perplexity on dataset split."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataloader.get_batch()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            outputs = model(x, targets=y)
        losses.append(outputs["loss"].item())
    model.train()
    mean_loss = float(np.mean(losses))
    perplexity = math.exp(min(mean_loss, 20.0))
    return mean_loss, perplexity


def save_checkpoint(
    checkpoint_dir: str,
    filename: str,
    model: Shortcake,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    config: ModelConfig,
):
    """Save training state checkpoint with param size tag in filename."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)

    # Format parameter count tag (e.g., 16.4m, 48.5m, 108.2m)
    num_params = raw_model.get_num_params()
    param_tag = f"{num_params / 1e6:.1f}m".replace(".0", "")
    if getattr(config, "jepa_enabled", False):
        param_tag += "_jepa"

    name_part, ext = os.path.splitext(filename)
    param_filename = f"{name_part}_{param_tag}{ext}"

    state = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": config,
        "param_tag": param_tag,
    }
    
    # Save parameter-tagged checkpoint once, then symlink alias to avoid double disk I/O
    param_filepath = os.path.join(checkpoint_dir, param_filename)
    torch.save(state, param_filepath)

    alias_filepath = os.path.join(checkpoint_dir, filename)
    if os.path.abspath(alias_filepath) != os.path.abspath(param_filepath):
        try:
            if os.path.lexists(alias_filepath):
                os.remove(alias_filepath)
            os.symlink(param_filename, alias_filepath)
        except OSError:
            torch.save(state, alias_filepath)
    print(f" Saved checkpoint to {param_filepath} (alias: {alias_filepath})", flush=True)


def configure_optimizers(
    model: Shortcake,
    learning_rate: float,
    weight_decay: float,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """Configures AdamW with weight decay applied ONLY to 2D matrix weights (protecting RMSNorm, biases, A_log, D)."""
    seen_param_ids = set()
    decay_params = []
    no_decay_params = []

    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (RMSNorm, nn.LayerNorm, nn.Embedding, nn.Conv1d)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if id(p) in seen_param_ids:
                continue
            seen_param_ids.add(id(p))

            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias") or "A_log" in pn or "D" in pn or "dt_proj" in fpn or isinstance(m, blacklist_weight_modules):
                no_decay_params.append(p)
            elif isinstance(m, whitelist_weight_modules):
                decay_params.append(p)
            else:
                no_decay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


def train(
    config: ModelConfig,
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
    max_steps: int = 10000,
    eval_interval: int = 100,
    log_interval: int = 10,
    save_interval: int = 500,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    learning_rate: float = 6e-4,
    min_lr: float = 6e-5,
    warmup_steps: int = 100,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    resume_checkpoint: Optional[str] = None,
    use_compile: bool = False,
):
    """Main training, validation, and test harness."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            num_cpus = os.cpu_count() or 2
            torch.set_num_threads(num_cpus)
            print(f"Starting Shortcake training on CPU ({num_cpus} threads)...", flush=True)
        else:
            print(f"Starting Shortcake training on CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

        use_bf16 = (device == "cuda") and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32)
        if HAS_MAMBA_CUDA and device == "cuda":
            scan_backend = "mamba-ssm CUDA kernel"
        elif HAS_TRITON and device == "cuda":
            scan_backend = "Triton fused CUDA selective scan"
        else:
            scan_backend = "chunked parallel PyTorch scan"
        conv_backend = "causal-conv1d CUDA kernel" if (HAS_CAUSAL_CONV1D and device == "cuda") else "PyTorch Conv1d"
        print(f"Precision: {amp_dtype} | SSM Backend: {scan_backend} | Conv Backend: {conv_backend}", flush=True)

        # Load Data Loaders
        train_loader = BinaryMemmapDataLoader(os.path.join(data_dir, "train.bin"), config.max_seq_len, batch_size, device)
        val_loader = BinaryMemmapDataLoader(os.path.join(data_dir, "val.bin"), config.max_seq_len, batch_size, device)
        test_loader = BinaryMemmapDataLoader(os.path.join(data_dir, "test.bin"), config.max_seq_len, batch_size, device)

        tokens_per_epoch = train_loader.total_tokens
        grad_accum_steps = max(1, grad_accum_steps)
        tokens_per_step = config.max_seq_len * batch_size * grad_accum_steps

        print(f"Dataset Loaded: {train_loader.total_tokens:,} training tokens.", flush=True)
        print(
            f"Batch Size: {batch_size} x Grad Accum: {grad_accum_steps} | "
            f"Tokens per Step: {tokens_per_step:,} | Est. Steps per Epoch: {max(1, tokens_per_epoch // tokens_per_step):,}",
            flush=True,
        )

        # Initialize Model
        model = Shortcake(config).to(device)
        print(f"Model initialized: {model.get_num_params():,} non-embedding parameters.\n", flush=True)

        # Optimizer with selective weight decay
        optimizer = configure_optimizers(
            model=model, learning_rate=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95)
        )

        start_step = 0
        best_val_loss = float("inf")

        # Resume from existing checkpoint if requested
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"Resuming training from checkpoint: {resume_checkpoint}", flush=True)
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt["step"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))

        if use_compile:
            print("Compiling model with torch.compile...", flush=True)
            model = torch.compile(model)

        # GradScaler only needed for FP16 (BF16 does not require loss scaling)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))

        model.train()
        start_time = time.time()
        step_start_time = time.time()

        print("--- Phase 1: Pre-training Harness Started ---", flush=True)
        if start_step == 0:
            val_loss0, val_ppl0 = evaluate(model, val_loader, eval_iters=10, device=device, amp_dtype=amp_dtype)
            print(f"Baseline (Step 0) | Val Loss: {val_loss0:.4f} | Val Perplexity: {val_ppl0:.2f}", flush=True)

        for step in range(start_step, max_steps + 1):
            lr = get_lr(step, warmup_steps, max_steps, learning_rate, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(grad_accum_steps):
                x, y = train_loader.get_batch()
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                    outputs = model(x, targets=y, compute_jepa=config.jepa_enabled)
                    micro_loss = outputs.get("total_loss", outputs["loss"])
                    scaled_loss = micro_loss / grad_accum_steps

                accum_loss += micro_loss.item() / grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if scaler.is_enabled():
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Calculate Epoch Progress
            current_tokens = step * tokens_per_step
            epoch = (current_tokens / tokens_per_epoch) + 1.0

            # Frequent logging every log_interval steps
            if step % log_interval == 0:
                dt = time.time() - step_start_time
                steps_logged = log_interval if step > start_step else 1
                sec_per_step = dt / steps_logged
                tok_per_sec = (tokens_per_step * steps_logged) / max(dt, 1e-5)
                print(
                    f"Epoch {epoch:.2f} | Step {step:5d}/{max_steps} | "
                    f"Train Loss: {accum_loss:.4f} | LR: {lr:.6f} | "
                    f"Speed: {tok_per_sec:.0f} tok/s ({sec_per_step:.2f}s/step)",
                    flush=True,
                )
                step_start_time = time.time()

            # Validation Evaluation Logging
            if step > 0 and (step % eval_interval == 0 or step == max_steps):
                val_loss, val_ppl = evaluate(model, val_loader, eval_iters=10, device=device, amp_dtype=amp_dtype)
                print(
                    f"\n>>> EVALUATION at Step {step}/{max_steps} (Epoch {epoch:.2f}) <<<\n"
                    f"    Train Loss: {accum_loss:.4f} | Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}\n",
                    flush=True,
                )

                # Save Best Model Checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(checkpoint_dir, "best_model.pt", model, optimizer, step, val_loss, config)

            # Periodic Checkpoint Saving
            if step > 0 and (step % save_interval == 0 or step == max_steps):
                save_checkpoint(checkpoint_dir, f"checkpoint_step_{step}.pt", model, optimizer, step, accum_loss, config)
                save_checkpoint(checkpoint_dir, "latest.pt", model, optimizer, step, accum_loss, config)

        # Final Test Evaluation
        print("\n--- Final Test Evaluation Harness ---", flush=True)
        test_loss, test_ppl = evaluate(model, test_loader, eval_iters=30, device=device, amp_dtype=amp_dtype)
        print(f" Final Test Loss: {test_loss:.4f} | Final Test Perplexity: {test_ppl:.2f}", flush=True)

    except Exception as e:
        print("\n CRITICAL ERROR DURING TRAINING:", flush=True)
        traceback.print_exc()
        raise e


def main():
    parser = argparse.ArgumentParser(description="Train Shortcake model with train/val/test harness & checkpoints.")
    parser.add_argument("--preset", type=str, default="small", help="Model size variant: 'tiny' (5M), 'small' (16M), 'base' (50M), 'medium' (110M), 'large' (350M).")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing binary dataset files.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps.")
    parser.add_argument("--eval_interval", type=int, default=100, help="Steps between validation evaluations.")
    parser.add_argument("--log_interval", type=int, default=1, help="Steps between frequent training logs.")
    parser.add_argument("--save_interval", type=int, default=500, help="Steps between checkpoint saves.")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per forward pass.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--jepa", action="store_true", help="Enable JEPA representation pre-training loss.")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile for faster training.")
    args = parser.parse_args()

    config = ModelConfig.from_preset(args.preset, jepa_enabled=args.jepa)
    train(
        config=config,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        resume_checkpoint=args.resume,
        use_compile=args.compile,
    )



if __name__ == "__main__":
    main()
