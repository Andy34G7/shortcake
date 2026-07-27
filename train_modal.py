import os
import modal

# Define Modal App
app = modal.App("shortcake-training")

# Persistent Volume for storing datasets & checkpoints across Modal GPU runs
volume = modal.Volume.from_name("shortcake-data-volume", create_if_missing=True)
VOLUME_DIR = "/vol"

# Define Modal GPU Container Image with PyTorch & HuggingFace dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "tokenizers>=0.15.0",
        "datasets>=2.16.0",
        "transformers>=4.36.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
    )
    .add_local_python_source("config", "mamba", "model", "train_tokenizer", "prepare_data", "train")
)


@app.function(image=image, volumes={VOLUME_DIR: volume}, timeout=3600)
def prepare_data_remote():
    """Runs tokenizer training and binary dataset tokenization directly on Modal."""
    from train_tokenizer import train_bpe_tokenizer
    from prepare_data import prepare_bin_data

    tokenizer_path = f"{VOLUME_DIR}/tokenizer.json"
    data_dir = f"{VOLUME_DIR}/data"
    os.makedirs(data_dir, exist_ok=True)

    print("=== Step 1: Training Byte-Level BPE Tokenizer on Modal ===")
    train_bpe_tokenizer(output_path=tokenizer_path, vocab_size=16384)


    print("=== Step 2: Preparing Memory-Mapped Binary Datasets on Modal ===")
    prepare_bin_data(tokenizer_path=tokenizer_path, output_dir=data_dir)

    volume.commit()
    print(" Data preparation complete and committed to Modal Volume!")


@app.function(
    image=image,
    gpu="A10G",  # Fast Nvidia A10G GPU (24GB VRAM). Options: "T4", "A10G", "L4", "A100"
    volumes={VOLUME_DIR: volume},
    timeout=86400,  # Up to 24 hours runtime limit
)
def train_remote(max_steps: int = 10000, batch_size: int = 16, lr: float = 6e-4):
    """Runs Shortcake model pre-training on Modal GPU with automatic AMP acceleration."""
    from config import ModelConfig
    from train import train

    data_dir = f"{VOLUME_DIR}/data"
    checkpoint_dir = f"{VOLUME_DIR}/checkpoints"

    if not os.path.exists(os.path.join(data_dir, "train.bin")):
        print("Data files not found in Modal volume. Triggering data preparation...")
        prepare_data_remote.local()

    config = ModelConfig()
    print("=== Starting Shortcake GPU Training on Modal ===")
    print(f"Max Steps: {max_steps:,} | Batch Size: {batch_size} | LR: {lr}")

    train(
        config=config,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        max_steps=max_steps,
        eval_interval=100,
        log_interval=10,
        save_interval=500,
        batch_size=batch_size,
        learning_rate=lr,
    )

    volume.commit()
    print(" Training complete! Checkpoints saved and committed to Modal Volume.")


@app.local_entrypoint()
def main(
    action: str = "train",
    max_steps: int = 10000,
    batch_size: int = 16,
):
    """Modal CLI local entrypoint."""
    if action == "prepare":
        print("Launching Data Preparation on Modal...")
        prepare_data_remote.remote()
    elif action == "train":
        print(f"Launching Training on Modal GPU (A10G)...")
        train_remote.remote(max_steps=max_steps, batch_size=batch_size)
    else:
        print("Invalid action! Use 'prepare' or 'train'.")
