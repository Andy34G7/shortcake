"""Score a Naver SPLADE baseline on the exact held-out pool train_splade.py evaluates on.

Same 1,000 docstring-stripped CodeSearchNet-python pairs, same MRR@10 / R@1 / R@10 math,
so the numbers sit next to ShortcakeSPLADE's directly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from train_splade import load_code_retrieval_pairs

CANDIDATES = ["naver/splade-v3", "naver/splade-cocondenser-ensembledistil"]


@torch.no_grad()
def encode(model, tok, texts, max_len, device):
    bs = 64 if device == "cuda" else 8
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i : i + bs], max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(**enc).logits
        vec = (torch.log1p(torch.relu(logits.float())) * enc["attention_mask"].unsqueeze(-1)).amax(dim=1)
        out.append(vec.cpu())
    return torch.cat(out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val = load_code_retrieval_pairs(max_train=1, max_val=1000)
    for name in CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForMaskedLM.from_pretrained(name).to(device).eval()
            break
        except Exception as e:  # gated repo without HF token -> fall back
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:160]}")
    else:
        raise SystemExit("no SPLADE baseline could be loaded")

    Q = encode(model, tok, [q for q, _ in val], 128, device)
    D = encode(model, tok, [d for _, d in val], 512, device)
    top = torch.topk(Q @ D.T, k=10, dim=-1).indices
    hit = top.eq(torch.arange(len(val)).unsqueeze(-1)).float()
    mrr = (hit / torch.arange(1, 11).float()).sum(-1).mean().item()
    print(
        f"BASELINE {name}: MRR@10={mrr:.4f} R@1={hit[:, 0].mean().item():.4f} "
        f"R@10={hit.any(-1).float().mean().item():.4f} "
        f"terms Q/D={(Q > 0).float().sum(-1).mean().item():.1f}/{(D > 0).float().sum(-1).mean().item():.1f}"
    )


if __name__ == "__main__":
    main()
