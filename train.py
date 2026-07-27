import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from config import ModelConfig
from model import Shortcake


class BinaryMemmapDataLoader:
    pass



class BinaryMemmapDataLoader:
    """High-throughput memory-mapped binary dataset iterator."""

    def __init__(self, filename: str, block_size: int, batch_size: int, device: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Dataset file {filename} does not exist. Run prepare_data.py first!")

        self.data = np.memmap(filename, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.total_tokens = len(self.data)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:

        """Fetch a random batch of input (x) and target (y) sequences."""
        ix = np.random.randint(0, len(self.data) - self.block_size, size=self.batch_size)
        x_stack = [self.data[i : i + self.block_size].astype(np.int64) for i in ix]
        y_stack = [self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix]

        x = torch.tensor(np.array(x_stack), dtype=torch.long, device=self.device)
        y = torch.tensor(np.array(y_stack), dtype=torch.long, device=self.device)
        return x, y


def get_lr(it: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(
    model: Shortcake,
    dataloader: BinaryMemmapDataLoader,
    eval_iters: int = 50,
) -> Tuple[float, float]:
    """Evaluate model loss and perplexity on dataset split."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataloader.get_batch()
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
    """Save training state checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": config,
    }
    torch.save(state, filepath)
    print(f" Saved checkpoint to {filepath}")



def train(
    config: ModelConfig,
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
    max_steps: int = 1000,
    eval_interval: int = 100,
    save_interval: int = 200,
    batch_size: int = 8,
    learning_rate: float = 6e-4,
    min_lr: float = 6e-5,
    warmup_steps: int = 100,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    resume_checkpoint: Optional[str] = None,
):
    """Main training, validation, and test harness."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # Optimize CPU threads for DigitalOcean 2-vCPU Droplet
        num_cpus = os.cpu_count() or 2
        torch.set_num_threads(num_cpus)
        print(f"Starting Shortcake (~20M) training on CPU ({num_cpus} threads)...")
    else:
        print(f"Starting Shortcake (~20M) training on CUDA device: {torch.cuda.get_device_name(0)}")


    # Load Data Loaders
    train_loader = BinaryMemmapDataLoader(os.path.join(data_dir, "train.bin"), config.max_seq_len, batch_size, device)
    val_loader = BinaryMemmapDataLoader(os.path.join(data_dir, "val.bin"), config.max_seq_len, batch_size, device)
    test_loader = BinaryMemmapDataLoader(os.path.join(data_dir, "test.bin"), config.max_seq_len, batch_size, device)

    # Initialize Model
    model = Shortcake(config).to(device)
    print(f"Model initialized: {model.get_num_params():,} non-embedding parameters.")


    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    start_step = 0
    best_val_loss = float("inf")

    # Resume from existing checkpoint if requested
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))

    # Scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    model.train()
    start_time = time.time()

    print("\n--- Phase 1: Pre-training Harness Started ---")
    for step in range(start_step, max_steps + 1):
        lr = get_lr(step, warmup_steps, max_steps, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = train_loader.get_batch()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(x, targets=y, compute_jepa=config.jepa_enabled)
            loss = outputs.get("total_loss", outputs["loss"])

        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Validation & Logging
        if step % eval_interval == 0 or step == max_steps:
            val_loss, val_ppl = evaluate(model, val_loader, eval_iters=20)
            dt = time.time() - start_time
            print(
                f"Step {step:5d}/{max_steps} | Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | LR: {lr:.6f} | Time: {dt:.1f}s"
            )
            start_time = time.time()

            # Save Best Model Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_dir, "best_model.pt", model, optimizer, step, val_loss, config)

        # Periodic Checkpoint Saving
        if step > 0 and (step % save_interval == 0 or step == max_steps):
            save_checkpoint(checkpoint_dir, f"checkpoint_step_{step}.pt", model, optimizer, step, loss.item(), config)
            save_checkpoint(checkpoint_dir, "latest.pt", model, optimizer, step, loss.item(), config)

    # Final Test Evaluation
    print("\n--- Final Test Evaluation Harness ---")
    test_loss, test_ppl = evaluate(model, test_loader, eval_iters=50)
    print(f" Final Test Loss: {test_loss:.4f} | Final Test Perplexity: {test_ppl:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train Smol-Hybrid-Coder model with train/val/test harness & checkpoints.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing binary dataset files.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--max_steps", type=int, default=500, help="Total training steps.")
    parser.add_argument("--eval_interval", type=int, default=50, help="Steps between validation evaluations.")
    parser.add_argument("--save_interval", type=int, default=100, help="Steps between checkpoint saves.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step.")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--jepa", action="store_true", help="Enable JEPA representation pre-training loss.")
    args = parser.parse_args()

    config = ModelConfig(jepa_enabled=args.jepa)
    train(
        config=config,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
