import os
import shutil
import torch
import torch.nn as nn
import numpy as np

from config import ModelConfig
from model import Shortcake
from mamba import MambaBlock, selective_scan_pytorch
from train_tokenizer import train_bpe_tokenizer
from prepare_data import prepare_bin_data, ShardWriter
from train import BinaryMemmapDataLoader, save_checkpoint, get_lr
from train_sft import tokenize_chatml_with_mask, sft_collate_fn
from generate import generate_code, sample_top_p_top_k
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


def test_model_presets():
    print("Testing Model Size Presets (tiny, small, base, medium, large)...")
    for name in ["tiny", "5m", "small", "16m", "base", "50m", "medium", "110m", "large", "350m"]:
        cfg = ModelConfig.from_preset(name)
        m = Shortcake(cfg)
        params = m.get_num_params()
        assert params > 0, f"Preset {name} parameter count invalid"

    print(" Model Size Presets test passed!")


def test_mamba_init_and_parallel_scan():
    print("Testing Mamba Initialization & Chunked Parallel Scan Equivalence...")
    config = ModelConfig(vocab_size=500, d_model=64, n_layer=4, n_head=4, d_state=16)
    model = Shortcake(config)

    # 1. Verify A matrix has multi-scale [1..d_state] per channel
    expected_row = torch.arange(1, config.d_state + 1, dtype=torch.float32)
    mamba_blocks = [m for m in model.modules() if isinstance(m, MambaBlock)]
    assert len(mamba_blocks) > 0
    for mb in mamba_blocks:
        A_vals = torch.exp(mb.A_log.detach())
        assert torch.allclose(A_vals, expected_row.unsqueeze(0).expand_as(A_vals), atol=1e-5), (
            "Mamba A_log must initialize each channel to [1, 2, ..., d_state]"
        )
        # Verify dt_proj.bias yields softplus in [0.001, 0.1]
        dt_bias_sp = torch.nn.functional.softplus(mb.dt_proj.bias.detach())
        assert dt_bias_sp.min() >= 0.0009 and dt_bias_sp.max() <= 0.105, "dt_proj.bias out of range"
        # Verify dt_proj.weight was not overwritten to std=0.02
        assert mb.dt_proj.weight.detach().abs().max() > 0.1, "dt_proj.weight appears overwritten"

    # 2. Verify chunked parallel scan matches naive sequential scan (forward + backward)
    torch.manual_seed(0)
    b, d, l, n = 2, 16, 75, 8  # l=75 is non-multiple of chunk_size=32
    u = torch.randn(b, d, l, requires_grad=True)
    delta = torch.nn.functional.softplus(torch.randn(b, d, l) - 2.0).detach().requires_grad_(True)
    A = (-torch.arange(1, n + 1, dtype=torch.float32).repeat(d, 1)).requires_grad_(True)
    B = torch.randn(b, n, l, requires_grad=True)
    C = torch.randn(b, n, l, requires_grad=True)
    D = torch.ones(d, requires_grad=True)

    # Reference sequential scan
    def ref_scan(u_, delta_, A_, B_, C_, D_):
        x = torch.zeros((b, d, n))
        outs = []
        for t in range(l):
            dt_t = delta_[:, :, t].unsqueeze(-1)
            u_t = u_[:, :, t].unsqueeze(-1)
            B_t = B_[:, :, t].unsqueeze(1)
            C_t = C_[:, :, t].unsqueeze(1)
            x = torch.exp(dt_t * A_.unsqueeze(0)) * x + (dt_t * u_t) * B_t
            outs.append((x * C_t).sum(dim=-1))
        y = torch.stack(outs, dim=-1) + u_ * D_.unsqueeze(-1)
        return y

    y_ref = ref_scan(u, delta, A, B, C, D)
    grads_ref = torch.autograd.grad(y_ref.sum(), [u, delta, A, B, C, D])

    y_par = selective_scan_pytorch(u, delta, A, B, C, D, chunk_size=32)
    grads_par = torch.autograd.grad(y_par.sum(), [u, delta, A, B, C, D])

    assert torch.allclose(y_par, y_ref, atol=1e-4, rtol=1e-4), "Parallel scan forward mismatch"
    for name, g_p, g_r in zip(["u", "delta", "A", "B", "C", "D"], grads_par, grads_ref):
        assert torch.allclose(g_p, g_r, atol=1e-4, rtol=1e-4), f"Parallel scan grad mismatch for {name}"

    print(" Mamba Initialization & Parallel Scan test passed!")


def test_sft_masking_and_tokenizer_decode():
    print("Testing SFT Assistant-Only Masking & Code Decode Preservation...")
    temp_dir = "test_temp_sft"
    os.makedirs(temp_dir, exist_ok=True)
    tokenizer_file = os.path.join(temp_dir, "tokenizer.json")

    from train_tokenizer import train_bpe_tokenizer_from_iterator
    sample_corpus = [
        "def add(a, b):\n    return a + b\n" * 10,
        "Hello, how can I help you today? Here is a Python function.",
    ]
    train_bpe_tokenizer_from_iterator(sample_corpus, output_path=tokenizer_file, vocab_size=500)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_file)

    # Verify code keywords & indentation survive default decode (skip_special_tokens=True)
    code = "def foo(x):\n    return x + 1"
    decoded = tok.decode(tok.encode(code).ids)
    assert "def " in decoded and "return " in decoded and "    " in decoded, (
        f"Code tokens stripped during decode: {decoded!r}"
    )

    # Verify SFT masking only supervises assistant turns
    messages = [
        {"role": "user", "content": "Write an add function."},
        {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
    ]
    sample = tokenize_chatml_with_mask(tok, messages, max_seq_len=512)
    assert sample is not None
    assert (sample["labels"] == -100).any(), "Prompt tokens must be masked with -100"
    assert (sample["labels"] != -100).any(), "Assistant tokens must be supervised"

    # Verify padded labels remain -100
    short_sample = {
        "input_ids": sample["input_ids"][:20],
        "labels": sample["labels"][:20],
    }
    inp, lbl = sft_collate_fn([sample, short_sample], pad_token_id=1)
    assert (lbl[1, 20:] == -100).all(), "Pad tokens must be masked with -100"

    shutil.rmtree(temp_dir)
    print(" SFT Masking & Code Decode test passed!")


def test_inference_cache_and_jepa_and_sampling():
    print("Testing SSM+KV Inference Cache Equivalence, JEPA Masking & Sampling Edge Cases...")
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=256, d_model=64, n_layer=4, n_head=4, d_state=8, max_seq_len=64, jepa_enabled=True)
    model = Shortcake(config).eval()

    # 1. Verify cached prefill + single-token decode matches full forward pass
    tokens = torch.randint(0, config.vocab_size, (2, 20))
    with torch.no_grad():
        full_logits = model(tokens)["logits"]

        cache = model.init_cache()
        prefill_logits = model(tokens[:, :12], cache=cache)["logits"]
        assert torch.allclose(prefill_logits, full_logits[:, :12], atol=1e-4, rtol=1e-4), "Prefill mismatch"

        for step in range(12, 20):
            step_logits = model(tokens[:, step : step + 1], cache=cache)["logits"]
            assert torch.allclose(step_logits, full_logits[:, step : step + 1], atol=1e-4, rtol=1e-4), (
                f"Cached decode mismatch at step {step}"
            )

    # 2. Verify JEPA predictor cannot leak target h[mask] (perturbing h[mask] does not change pred_h[mask])
    h = torch.randn(2, 16, config.d_model)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, [3, 7, 11]] = True
    h_perturbed = h.clone()
    h_perturbed[mask] = h_perturbed[mask] + 100.0
    with torch.no_grad():
        pred1 = model.jepa_predictor(h, mask)
        pred2 = model.jepa_predictor(h_perturbed, mask)
    assert torch.allclose(pred1, pred2, atol=1e-5), "JEPA predictor leaked masked target tokens"

    # 3. Verify temperature=0 greedy decoding & repetition penalty
    logits = torch.tensor([[1.0, 5.0, -2.0, 3.0]])
    seen = torch.tensor([[1, 2]])
    greedy = sample_top_p_top_k(logits, seen, temperature=0.0, repetition_penalty=1.0)
    assert greedy.item() == 1, "Greedy decoding should select argmax"
    penalized = sample_top_p_top_k(logits, seen, temperature=0.0, repetition_penalty=2.0)
    assert penalized.item() == 3, "Repetition penalty should demote token 1 (5/2=2.5 < 3.0)"

    # 4. Verify get_lr handles max_steps <= warmup_steps without crashing
    lr_val = get_lr(it=15, warmup_steps=100, max_steps=10, max_lr=1e-3, min_lr=1e-4)
    assert lr_val > 0

    # 5. Verify ShardWriter + multi-shard BinaryMemmapDataLoader auto-discovery
    shard_dir = "test_temp_shards"
    os.makedirs(shard_dir, exist_ok=True)
    writer = ShardWriter(shard_dir, prefix="train", shard_size=100)
    writer.add(np.arange(250, dtype=np.uint16))
    shards = writer.finalize()
    assert len(shards) == 3, f"Expected 3 shards, got {len(shards)}"
    loader = BinaryMemmapDataLoader(os.path.join(shard_dir, "train.bin"), block_size=16, batch_size=4, device="cpu")
    assert loader.total_tokens == 250 and len(loader.shards) == 3
    xb, yb = loader.get_batch()
    assert xb.shape == (4, 16) and yb.shape == (4, 16)
    shutil.rmtree(shard_dir)

    print(" Inference Cache, JEPA Masking, Shards & Sampling tests passed!")


def main():
    print("=== Running Shortcake Test Suite ===")
    test_model_forward_backward()
    test_tokenizer_and_data_prep()
    test_overfit_mini_batch()
    test_splade_head()
    test_model_presets()
    test_mamba_init_and_parallel_scan()
    test_sft_masking_and_tokenizer_decode()
    test_inference_cache_and_jepa_and_sampling()
    print("\n ALL TESTS PASSED SUCCESSFULLY!")



if __name__ == "__main__":
    main()
