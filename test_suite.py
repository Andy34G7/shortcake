import os
import shutil
import torch
import torch.nn as nn
import numpy as np

from config import ModelConfig
from model import Shortcake
from mamba import MambaBlock
from train_tokenizer import train_bpe_tokenizer
from prepare_data import prepare_bin_data
from train import BinaryMemmapDataLoader, save_checkpoint
from generate import generate_code
from splade_head import ShortcakeSPLADE


def test_model_forward_backward():
    print("Testing Model Forward and Backward Pass...")
    config = ModelConfig(vocab_size=1000, d_model=128, n_layer=4, n_head=4, max_seq_len=64)
    model = Shortcake(config)

    x = torch.randint(0, config.vocab_size, (2, 32))
    y = torch.randint(0, config.vocab_size, (2, 32))

    output = model(x, targets=y, compute_jepa=True)
    loss = output.get("total_loss", output["loss"])

    assert loss is not None, "Loss should not be None"
    assert output["logits"].shape == (2, 32, config.vocab_size), "Logits shape mismatch"

    loss.backward()

    # Verify gradients flow to embeddings and layers
    assert model.tok_embeddings.weight.grad is not None, "Gradient missing on tok_embeddings"
    print(" Model Forward and Backward Pass test passed!")


def test_tokenizer_and_data_prep():
    print("Testing Tokenizer Training & Binary Data Preparation...")
    temp_dir = "test_temp_data"
    os.makedirs(temp_dir, exist_ok=True)

    sample_text_file = os.path.join(temp_dir, "code.txt")
    tokenizer_file = os.path.join(temp_dir, "tokenizer.json")

    sample_text = "def add(a, b):\n    return a + b\n" * 20
    with open(sample_text_file, "w", encoding="utf-8") as f:
        f.write(sample_text)

    from train_tokenizer import train_bpe_tokenizer_from_iterator
    train_bpe_tokenizer_from_iterator([sample_text], output_path=tokenizer_file, vocab_size=500)
    assert os.path.exists(tokenizer_file), "Tokenizer file was not created"

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_file)
    enc = tok.encode(sample_text)
    arr = np.array(enc.ids, dtype=np.uint16)

    train_filename = os.path.join(temp_dir, "train.bin")
    val_filename = os.path.join(temp_dir, "val.bin")
    test_filename = os.path.join(temp_dir, "test.bin")

    arr[:100].tofile(train_filename)
    arr[100:120].tofile(val_filename)
    arr[120:].tofile(test_filename)

    loader = BinaryMemmapDataLoader(train_filename, block_size=16, batch_size=2, device="cpu")
    x, y = loader.get_batch()
    assert x.shape == (2, 16), "DataLoader batch shape mismatch"
    assert y.shape == (2, 16), "DataLoader target shape mismatch"

    shutil.rmtree(temp_dir)
    print(" Tokenizer & Data Prep test passed!")



def test_overfit_mini_batch():
    print("Testing Overfit Mini-Batch Loss Decrease...")
    config = ModelConfig(vocab_size=500, d_model=128, n_layer=4, n_head=4, max_seq_len=64)
    model = Shortcake(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))

    initial_loss = model(x, targets=y)["loss"].item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = model(x, targets=y)["loss"]
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"Initial loss: {initial_loss:.4f} -> Final loss after 30 steps: {final_loss:.4f}")
    assert final_loss < initial_loss, "Model failed to reduce loss during overfit test"
    print(" Overfit Mini-Batch test passed!")


def test_splade_head():
    print("Testing SPLADE Head Sparse Representation...")
    config = ModelConfig(vocab_size=1000, d_model=128, n_layer=4, n_head=4)
    backbone = Shortcake(config)
    splade_model = ShortcakeSPLADE(backbone)

    x = torch.randint(0, config.vocab_size, (2, 16))
    out = splade_model(x)

    sparse_vec = out["sparse_vector"]
    assert sparse_vec.shape == (2, config.vocab_size), "SPLADE vector shape mismatch"
    assert torch.all(sparse_vec >= 0), "SPLADE weights must be non-negative"
    print(" SPLADE Head test passed!")


def main():
    print("=== Running Shortcake Test Suite ===")
    test_model_forward_backward()
    test_tokenizer_and_data_prep()
    test_overfit_mini_batch()
    test_splade_head()
    print("\n ALL TESTS PASSED SUCCESSFULLY!")



if __name__ == "__main__":
    main()
