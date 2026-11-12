import os
import sys
import math
import types
import pickle
import shutil
import importlib
import torch
import torch.nn as nn
import numpy as np

from config import ModelConfig
from model import Shortcake, AttentionBlock, jepa_span_mask
from mamba import MambaBlock
from train import BinaryMemmapDataLoader, save_checkpoint, configure_optimizers
from generate import generate_code

try:
    from splade_head import ShortcakeSPLADE
except ImportError:  # optional module, not present in every checkout
    ShortcakeSPLADE = None


class SkipTest(Exception):
    """Raised by a test that cannot run in this environment."""


def _import_without_datasets(module_name: str):
    """Import a module whose top-level `from datasets import load_dataset` the test does not need.

    train_tokenizer.py and prepare_data.py import Hugging Face `datasets` at module level only
    for their streaming entry points. If it is not installed, a placeholder is put in
    sys.modules for the duration of the import; calling load_dataset on it raises.
    """
    try:
        import datasets  # noqa: F401
        return importlib.import_module(module_name)
    except ImportError:
        pass

    def _missing(*args, **kwargs):
        raise ImportError("`datasets` is not installed")

    stub = types.ModuleType("datasets")
    stub.load_dataset = _missing
    sys.modules["datasets"] = stub
    try:
        return importlib.import_module(module_name)
    finally:
        sys.modules.pop("datasets", None)


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

    train_tokenizer = _import_without_datasets("train_tokenizer")
    train_tokenizer.train_bpe_tokenizer_from_iterator([sample_text], output_path=tokenizer_file, vocab_size=500)
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


def _overfit(jepa: bool):
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=500, d_model=128, n_layer=4, n_head=4, max_seq_len=64, jepa_enabled=jepa)
    model = Shortcake(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, config.vocab_size, (2, 32))
    y = torch.randint(0, config.vocab_size, (2, 32))

    initial = model(x, targets=y, compute_jepa=jepa)
    initial_loss = initial["loss"].item()
    initial_jepa = initial["jepa_loss"].item() if jepa else None
    for _ in range(30):
        optimizer.zero_grad()
        out = model(x, targets=y, compute_jepa=jepa)
        loss = out.get("total_loss", out["loss"])
        loss.backward()
        optimizer.step()

    final_loss = out["loss"].item()
    msg = f"[jepa={jepa}] Initial LM loss: {initial_loss:.4f} -> Final LM loss after 30 steps: {final_loss:.4f}"
    if jepa:
        msg += f" | JEPA loss {initial_jepa:.4f} -> {out['jepa_loss'].item():.4f}"
    print(msg)
    assert math.isfinite(final_loss), "Non-finite loss during overfit test"
    assert final_loss < initial_loss, "Model failed to reduce loss during overfit test"


def test_overfit_mini_batch():
    print("Testing Overfit Mini-Batch Loss Decrease...")
    _overfit(jepa=False)
    print(" Overfit Mini-Batch test passed!")


def test_overfit_mini_batch_jepa():
    print("Testing Overfit Mini-Batch Loss Decrease with JEPA...")
    _overfit(jepa=True)
    print(" Overfit Mini-Batch (JEPA) test passed!")


def test_splade_head():
    print("Testing SPLADE Head Sparse Representation...")
    if ShortcakeSPLADE is None:
        raise SkipTest("splade_head module not found")
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
        assert cfg.head_dim % 2 == 0, f"Preset {name} has odd head_dim (RoPE needs even)"
        m = Shortcake(cfg)
        params = m.get_num_params()
        assert params > 0, f"Preset {name} parameter count invalid"

    print(" Model Size Presets test passed!")


def test_rope_param_count_and_state_dict():
    print("Testing RoPE adds no parameters or state_dict entries...")
    kwargs = dict(vocab_size=500, d_model=128, n_layer=4, n_head=4, max_seq_len=64)
    with_rope = Shortcake(ModelConfig(rope=True, **kwargs))
    without_rope = Shortcake(ModelConfig(rope=False, **kwargs))
    assert with_rope.get_num_params() == without_rope.get_num_params(), "RoPE changed parameter count"
    sd_a, sd_b = with_rope.state_dict(), without_rope.state_dict()
    assert sd_a.keys() == sd_b.keys(), "RoPE changed state_dict keys"
    assert all(sd_a[k].shape == sd_b[k].shape for k in sd_a), "RoPE changed state_dict shapes"
    without_rope.load_state_dict(sd_a)  # strict load across the two variants
    print(" RoPE parameter count test passed!")


def test_rope_position_dependence():
    print("Testing attention output depends on position with RoPE...")
    torch.manual_seed(0)
    kwargs = dict(vocab_size=500, d_model=64, n_layer=4, n_head=4, max_seq_len=16)
    block_rope = AttentionBlock(ModelConfig(rope=True, **kwargs)).eval()
    block_plain = AttentionBlock(ModelConfig(rope=False, **kwargs)).eval()
    block_plain.load_state_dict(block_rope.state_dict())

    # Without positional information, the last position attends to the *set* of earlier
    # positions, so permuting them leaves its output unchanged. RoPE must break that.
    x = torch.randn(1, 12, 64)
    perm = torch.cat([torch.randperm(11), torch.tensor([11])])
    x_perm = x[:, perm]
    with torch.no_grad():
        plain_a, plain_b = block_plain(x)[:, -1], block_plain(x_perm)[:, -1]
        rope_a, rope_b = block_rope(x)[:, -1], block_rope(x_perm)[:, -1]
    assert torch.allclose(plain_a, plain_b, atol=1e-5), \
        "Sanity check failed: attention without RoPE should be order-invariant here"
    diff = (rope_a - rope_b).abs().max().item()
    assert diff > 1e-4, f"RoPE attention output does not depend on position (max diff {diff})"

    # Longer than max_seq_len: the cos/sin table extends lazily, and the prefix agrees.
    long_x = torch.randn(2, 40, 64)
    with torch.no_grad():
        out_long = block_rope(long_x)
        out_prefix = block_rope(long_x[:, :10])
    assert out_long.shape == long_x.shape
    assert block_rope.rotary.cos_cached.shape[0] >= 40
    assert torch.allclose(out_long[:, :10], out_prefix, atol=1e-5), "Extended RoPE table disagrees with prefix"
    print(" RoPE position dependence test passed!")


def test_init_scales():
    print("Testing dt_proj and residual projection initialization...")
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=500, d_model=256, n_layer=8, n_head=4, max_seq_len=64, jepa_enabled=True)
    model = Shortcake(config)

    bound = config.dt_rank ** -0.5
    expected_std = bound / math.sqrt(3.0)  # std of U(-bound, bound)
    residual_std = 0.02 / math.sqrt(2 * config.n_layer)
    n_mamba = 0
    for m in model.modules():
        if isinstance(m, MambaBlock):
            n_mamba += 1
            w = m.dt_proj.weight
            assert w.abs().max().item() <= bound + 1e-6, "dt_proj.weight exceeds uniform bound"
            assert abs(w.std().item() - expected_std) / expected_std < 0.1, \
                f"dt_proj.weight std {w.std().item():.4f}, expected ~{expected_std:.4f} (not 0.02)"
            dt = torch.nn.functional.softplus(m.dt_proj.bias)
            assert dt.min().item() >= 0.001 - 1e-6 and dt.max().item() <= 0.1 + 1e-6, "dt_proj.bias init lost"
            assert abs(m.out_proj.weight.std().item() - residual_std) / residual_std < 0.1, "Mamba out_proj std"
        elif isinstance(m, AttentionBlock):
            assert abs(m.out_proj.weight.std().item() - residual_std) / residual_std < 0.1, "Attention out_proj std"
            assert abs(m.mlp.w2.weight.std().item() - residual_std) / residual_std < 0.1, "MLP w2 std"
            assert abs(m.qkv_proj.weight.std().item() - 0.02) / 0.02 < 0.1, "qkv_proj should keep std 0.02"
    assert n_mamba > 0
    # The JEPA predictor is not part of the residual stream and keeps the plain 0.02 init.
    pred_out = model.jepa_predictor.predictor_layer.self_attn.out_proj.weight
    assert abs(pred_out.std().item() - 0.02) / 0.02 < 0.1, "JEPA predictor out_proj should not be rescaled"
    print(" Init scale test passed!")


def test_jepa_objective():
    print("Testing JEPA objective (loss, gradients, mask, clean-pass logits)...")
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=500, d_model=128, n_layer=4, n_head=4, max_seq_len=256, jepa_enabled=True, dropout=0.1
    )
    model = Shortcake(config)
    x = torch.randint(0, config.vocab_size, (4, 256))
    y = torch.randint(0, config.vocab_size, (4, 256))

    out = model(x, targets=y, compute_jepa=True)
    jepa_loss = out["jepa_loss"]
    assert torch.isfinite(jepa_loss) and torch.isfinite(out["total_loss"]), "JEPA loss not finite"
    expected_total = out["loss"] + config.jepa_loss_weight * jepa_loss
    assert torch.allclose(out["total_loss"], expected_total), "total_loss != lm + weight * jepa"

    # Masked fraction close to the ratio, and the mask is made of contiguous spans.
    mask = out["jepa_mask"]
    frac = mask.float().mean().item()
    assert abs(frac - config.jepa_mask_ratio) < 0.02, f"Masked fraction {frac:.3f} vs ratio {config.jepa_mask_ratio}"
    starts = (mask[:, 1:] & ~mask[:, :-1]).sum(dim=1) + mask[:, 0].long()
    max_spans = math.ceil(round(config.jepa_mask_ratio * 256) / config.jepa_span_len)
    assert int(starts.max()) <= max_spans, "Mask is not span-structured"

    # Gradients of the JEPA loss alone reach the backbone (via the context pass) and the mask embedding.
    model.zero_grad()
    jepa_loss.backward()
    assert model.jepa_mask_embedding.grad is not None and model.jepa_mask_embedding.grad.abs().sum() > 0, \
        "No gradient on jepa_mask_embedding"
    for i, layer in enumerate(model.layers):
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        assert grads and sum(g.abs().sum() for g in grads) > 0, f"No JEPA gradient reached backbone layer {i}"

    # The predictor never sees the clean pass: the target is detached, so with compute_jepa the
    # LM logits are exactly those of a plain forward (same seed covers dropout, train mode).
    for train_mode in (True, False):
        model.train(train_mode)
        torch.manual_seed(123)
        logits_plain = model(x)["logits"]
        torch.manual_seed(123)
        logits_jepa = model(x, targets=y, compute_jepa=True)["logits"]
        assert torch.equal(logits_plain, logits_jepa), f"compute_jepa changed LM logits (train={train_mode})"

    # Helper edge cases: exact counts, short sequences.
    m = jepa_span_mask(3, 32, 0.15, 8, device=torch.device("cpu"))
    assert (m.sum(dim=1) == round(0.15 * 32)).all()
    assert jepa_span_mask(2, 1, 0.15, 8, device=torch.device("cpu")) is None
    print(f" JEPA objective test passed! (masked fraction {frac:.3f})")


def test_jepa_mask_embedding_no_weight_decay():
    print("Testing JEPA mask embedding is excluded from weight decay...")
    model = Shortcake(ModelConfig(vocab_size=500, d_model=64, n_layer=4, n_head=4, jepa_enabled=True))
    optimizer = configure_optimizers(model, learning_rate=1e-3, weight_decay=0.1)
    no_decay = {id(p) for g in optimizer.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    assert id(model.jepa_mask_embedding) in no_decay, "jepa_mask_embedding is in the weight-decay group"
    n_opt = sum(len(g["params"]) for g in optimizer.param_groups)
    assert n_opt == len(list(model.parameters())), "Optimizer does not cover every parameter exactly once"
    print(" JEPA no-decay test passed!")


def test_causality():
    print("Testing causality (future tokens do not change earlier logits)...")
    for rope in (True, False):
        torch.manual_seed(0)
        config = ModelConfig(vocab_size=500, d_model=64, n_layer=4, n_head=4, max_seq_len=32, rope=rope)
        model = Shortcake(config).eval()
        x = torch.randint(0, config.vocab_size, (2, 24))
        t = 15
        x2 = x.clone()
        x2[:, t] = (x2[:, t] + 1) % config.vocab_size
        with torch.no_grad():
            a = model(x)["logits"]
            b = model(x2)["logits"]
        assert torch.allclose(a[:, :t], b[:, :t], atol=1e-5), f"Earlier logits changed (rope={rope})"
        assert not torch.allclose(a[:, t:], b[:, t:], atol=1e-5), f"Later logits did not change (rope={rope})"
    print(" Causality test passed!")


def test_legacy_checkpoint_compat():
    print("Testing old pickled configs / checkpoints keep working...")
    # A config pickled before `rope`, `jepa_span_len` and `jepa_loss_weight` existed.
    config = ModelConfig(vocab_size=500, d_model=64, n_layer=4, n_head=4, jepa_enabled=True)
    for name in ("rope", "jepa_span_len", "jepa_loss_weight"):
        del config.__dict__[name]
    old = pickle.loads(pickle.dumps(config))
    assert old.rope is False, "Old configs must keep their pre-RoPE behaviour"
    assert pickle.loads(pickle.dumps(ModelConfig())).rope is True, "New configs must keep rope=True"

    model = Shortcake(old)
    assert all(b.rotary is None for b in model.modules() if isinstance(b, AttentionBlock))
    out = model(torch.randint(0, 500, (2, 32)), targets=torch.randint(0, 500, (2, 32)), compute_jepa=True)
    assert torch.isfinite(out["total_loss"])

    # A state_dict from the earlier JEPA variant has no mask embedding; strict loading still works.
    sd = {k: v for k, v in model.state_dict().items() if k != "jepa_mask_embedding"}
    Shortcake(old).load_state_dict(sd)
    print(" Legacy checkpoint compatibility test passed!")


def main():
    print("=== Running Shortcake Test Suite ===")
    tests = [
        test_model_forward_backward,
        test_tokenizer_and_data_prep,
        test_overfit_mini_batch,
        test_overfit_mini_batch_jepa,
        test_splade_head,
        test_rope_param_count_and_state_dict,
        test_rope_position_dependence,
        test_init_scales,
        test_jepa_objective,
        test_jepa_mask_embedding_no_weight_decay,
        test_causality,
        test_legacy_checkpoint_compat,
        test_model_presets,
    ]
    skipped = []
    for test in tests:
        try:
            test()
        except SkipTest as e:
            print(f" SKIPPED {test.__name__}: {e}")
            skipped.append(test.__name__)

    if skipped:
        print(f"\n ALL TESTS PASSED ({len(skipped)} skipped: {', '.join(skipped)})")
    else:
        print("\n ALL TESTS PASSED SUCCESSFULLY!")



if __name__ == "__main__":
    main()
