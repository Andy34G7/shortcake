import os
import time
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

from config import ModelConfig
from model import Shortcake
from train import configure_optimizers, get_lr, save_checkpoint


CHATML_SYSTEM_PROMPT = "<|im_start|>system\nYou are Shortcake, a helpful AI assistant.<|im_end|>\n"


def format_chatml(messages: List[Dict[str, str]]) -> str:
    """Formats a list of chat messages into standard ChatML string."""
    formatted = CHATML_SYSTEM_PROMPT
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted


def tokenize_chatml_with_mask(
    tokenizer: Tokenizer,
    messages: List[Dict[str, str]],
    max_seq_len: int = 1024,
) -> Optional[Dict[str, torch.Tensor]]:
    """Tokenizes a ChatML conversation and masks non-assistant tokens with -100."""
    input_ids: List[int] = []
    labels: List[int] = []

    sys_ids = tokenizer.encode(CHATML_SYSTEM_PROMPT).ids
    input_ids.extend(sys_ids)
    labels.extend([-100] * len(sys_ids))

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix_ids = tokenizer.encode(f"<|im_start|>{role}\n").ids
        body_ids = tokenizer.encode(f"{content}<|im_end|>\n").ids

        input_ids.extend(prefix_ids)
        labels.extend([-100] * len(prefix_ids))

        input_ids.extend(body_ids)
        if role == "assistant":
            labels.extend(body_ids)
        else:
            labels.extend([-100] * len(body_ids))

    if len(input_ids) < 16 or len(input_ids) > max_seq_len:
        return None
    if not any(lbl != -100 for lbl in labels[1:]):
        return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def prepare_sft_dataset(
    tokenizer: Tokenizer,
    dataset_name: str = "HuggingFaceTB/smoltalk",
    split_name: str = "all",
    max_samples: int = 20000,
    max_seq_len: int = 1024,
) -> List[Dict[str, torch.Tensor]]:
    """Loads SFT instruction dataset and tokenizes into ChatML input_ids & assistant-only labels."""
    print(f"Streaming instruction dataset '{dataset_name}' (split: {split_name})...")
    ds = load_dataset(dataset_name, name=split_name, split="train", streaming=True)

    samples = []
    count = 0

    for item in tqdm(ds, total=max_samples, desc="Processing SFT Data"):
        messages = item.get("messages", [])
        if not messages:
            continue

        sample = tokenize_chatml_with_mask(tokenizer, messages, max_seq_len=max_seq_len)
        if sample is None:
            continue

        samples.append(sample)
        count += 1
        if count >= max_samples:
            break

    print(f" Collected {len(samples):,} tokenized SFT instruction samples.")
    return samples


def sft_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 1):
    """Pads SFT batch sequences dynamically."""
    max_len = max(sample["input_ids"].size(0) for sample in batch)
    
    b_input_ids = []
    b_labels = []

    for sample in batch:
        inp = sample["input_ids"]
        lbl = sample["labels"]
        pad_len = max_len - inp.size(0)

        padded_inp = torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)]) # -100 ignores loss on pad

        b_input_ids.append(padded_inp)
        b_labels.append(padded_lbl)

    return torch.stack(b_input_ids), torch.stack(b_labels)


@torch.no_grad()
def evaluate_sft(
    model: Shortcake,
    val_data: List[Dict[str, torch.Tensor]],
    batch_size: int,
    pad_token_id: int,
    device: str,
    amp_dtype: torch.dtype,
    max_batches: int = 20,
) -> float:
    """Evaluate SFT loss on held-out validation samples (assistant tokens only)."""
    model.eval()
    losses = []
    for i in range(0, min(len(val_data), max_batches * batch_size), batch_size):
        batch_samples = val_data[i : i + batch_size]
        if not batch_samples:
            break
        input_ids, labels = sft_collate_fn(batch_samples, pad_token_id=pad_token_id)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            outputs = model(input_ids[:, :-1], targets=labels[:, 1:])
        losses.append(outputs["loss"].item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def train_sft(
    base_checkpoint: str = "checkpoints/best_model.pt",
    tokenizer_path: str = "tokenizer.json",
    output_dir: str = "checkpoints_sft",
    max_steps: int = 2000,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Supervised Fine-Tuning (SFT) runner for Shortcake model."""
    if not os.path.exists(base_checkpoint):
        raise FileNotFoundError(f"Pre-trained base model checkpoint not found at {base_checkpoint}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        pad_token_id = 1

    print(f"=== Starting Shortcake Supervised Fine-Tuning (SFT) ===")
    print(f"Base Checkpoint: {base_checkpoint} | Device: {device} | Max Steps: {max_steps:,}")

    # Load Base Model Checkpoint
    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", ModelConfig())
    model = Shortcake(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded base model: {model.get_num_params():,} parameters.")

    # Prepare SFT dataset and hold out 5% for validation
    sft_data = prepare_sft_dataset(tokenizer, max_samples=max_steps * batch_size, max_seq_len=config.max_seq_len)
    val_size = max(batch_size, int(len(sft_data) * 0.05))
    val_data = sft_data[:val_size]
    train_data = sft_data[val_size:] if len(sft_data) > val_size else sft_data

    # Configure Optimizer with lower learning rate for fine-tuning
    optimizer = configure_optimizers(model=model, learning_rate=learning_rate, weight_decay=0.01)

    use_bf16 = (device == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))

    warmup_steps = max(1, min(100, max_steps // 10))
    min_lr = learning_rate * 0.1

    model.train()
    step = 0
    num_samples = len(train_data)
    best_val_loss = float("inf")

    start_time = time.time()

    while step < max_steps:
        # Shuffle dataset each epoch
        indices = torch.randperm(num_samples)
        for i in range(0, max(1, num_samples - batch_size + 1), batch_size):
            if step >= max_steps:
                break

            lr = get_lr(step, warmup_steps, max_steps, learning_rate, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            batch_indices = indices[i : i + batch_size]
            batch_samples = [train_data[idx] for idx in batch_indices]
            input_ids, labels = sft_collate_fn(batch_samples, pad_token_id=pad_token_id)

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass: supervise ONLY assistant tokens (labels masks pads & prompts with -100)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                outputs = model(input_ids[:, :-1], targets=labels[:, 1:])
                loss = outputs["loss"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            step += 1

            if step % 20 == 0 or step == max_steps:
                dt = time.time() - start_time
                tok_per_sec = (step * batch_size * input_ids.size(1)) / max(dt, 1e-5)
                print(
                    f"SFT Step {step:4d}/{max_steps} | Loss: {loss.item():.4f} | "
                    f"LR: {lr:.6f} | Speed: {tok_per_sec:.0f} tok/s"
                )

            if step % 500 == 0 or step == max_steps:
                val_loss = evaluate_sft(model, val_data, batch_size, pad_token_id, device, amp_dtype)
                print(f">>> SFT Eval at Step {step}/{max_steps} | Val Loss: {val_loss:.4f}")
                save_checkpoint(output_dir, f"sft_model_step_{step}.pt", model, optimizer, step, val_loss, config)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(output_dir, "best_sft_model.pt", model, optimizer, step, val_loss, config)

    print(f"\n Supervised Fine-Tuning Complete! Best val loss: {best_val_loss:.4f} (saved to '{output_dir}/best_sft_model.pt').")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shortcake Supervised Fine-Tuning (SFT)")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to base pretrained model.")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max SFT steps.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    args = parser.parse_args()

    train_sft(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
