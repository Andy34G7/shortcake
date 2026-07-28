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
from train import configure_optimizers, save_checkpoint


CHATML_SYSTEM_PROMPT = "<|im_start|>system\nYou are Shortcake, a helpful AI assistant.<|im_end|>\n"

def format_chatml(messages: List[Dict[str, str]]) -> str:
    """Formats a list of chat messages into standard ChatML string."""
    formatted = CHATML_SYSTEM_PROMPT
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted


def prepare_sft_dataset(
    tokenizer: Tokenizer,
    dataset_name: str = "HuggingFaceTB/smoltalk",
    split_name: str = "all",
    max_samples: int = 20000,
    max_seq_len: int = 1024,
) -> List[Dict[str, torch.Tensor]]:
    """Loads SFT instruction dataset and tokenizes into ChatML input_ids & masked labels."""
    print(f"Streaming instruction dataset '{dataset_name}' (split: {split_name})...")
    ds = load_dataset(dataset_name, name=split_name, split="train", streaming=True)
    
    samples = []
    count = 0
    
    for item in tqdm(ds, total=max_samples, desc="Processing SFT Data"):
        messages = item.get("messages", [])
        if not messages:
            continue

        full_chatml = format_chatml(messages)
        encoding = tokenizer.encode(full_chatml)
        input_ids = encoding.ids

        if len(input_ids) < 16 or len(input_ids) > max_seq_len:
            continue

        # Prepare target labels (calculate loss on full sequence)
        labels = list(input_ids)

        samples.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })
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

    print(f"=== Starting Shortcake Supervised Fine-Tuning (SFT) ===")
    print(f"Base Checkpoint: {base_checkpoint} | Device: {device} | Max Steps: {max_steps:,}")

    # Load Base Model Checkpoint
    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", ModelConfig())
    model = Shortcake(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded base model: {model.get_num_params():,} parameters.")

    # Prepare SFT dataset
    sft_data = prepare_sft_dataset(tokenizer, max_samples=max_steps * batch_size, max_seq_len=config.max_seq_len)
    
    # Configure Optimizer with lower learning rate for fine-tuning
    optimizer = configure_optimizers(model=model, learning_rate=learning_rate, weight_decay=0.01)

    model.train()
    step = 0
    num_samples = len(sft_data)
    
    start_time = time.time()
    
    while step < max_steps:
        # Shuffle dataset each epoch
        indices = torch.randperm(num_samples)
        for i in range(0, num_samples - batch_size, batch_size):
            if step >= max_steps:
                break

            batch_indices = indices[i : i + batch_size]
            batch_samples = [sft_data[idx] for idx in batch_indices]
            input_ids, labels = sft_collate_fn(batch_samples, pad_token_id=1)
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids[:, :-1], targets=input_ids[:, 1:])
            loss = outputs["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1

            if step % 20 == 0 or step == max_steps:
                dt = time.time() - start_time
                tok_per_sec = (step * batch_size * input_ids.size(1)) / max(dt, 1e-5)
                print(f"SFT Step {step:4d}/{max_steps} | Loss: {loss.item():.4f} | Speed: {tok_per_sec:.0f} tok/s")

            if step % 500 == 0 or step == max_steps:
                sft_ckpt_path = os.path.join(output_dir, f"sft_model_step_{step}.pt")
                save_checkpoint(output_dir, "best_sft_model.pt", model, optimizer, step, loss.item(), config)
                print(f" Saved SFT Checkpoint: {sft_ckpt_path}")

    print("\n Supervised Fine-Tuning Complete! Best checkpoint saved to 'checkpoints_sft/best_sft_model.pt'.")


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
