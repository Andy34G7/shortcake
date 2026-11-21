import os
import re
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

from config import ModelConfig
from model import Shortcake
from splade_head import ShortcakeSPLADE
from train import configure_optimizers, get_lr


QUERY_SUFFIX = "\nQuery keywords:"
DOC_SUFFIX = "\nCode keywords:"


def strip_docstring(code: str, docstring: str) -> str:
    """Remove docstring from code body so retrieval cannot cheat via verbatim docstring overlap."""
    cleaned = code.replace(docstring, "")
    cleaned = re.sub(r'"""[\s\S]*?"""', "", cleaned)
    cleaned = re.sub(r"'''[\s\S]*?'''", "", cleaned)
    return cleaned.strip()


def load_code_retrieval_pairs(
    dataset_name: str = "Nan-Do/code-search-net-python",
    max_train: int = 50000,
    max_val: int = 1000,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Stream (docstring, pure_code) pairs with docstrings stripped from code bodies."""
    print(f"Streaming retrieval dataset '{dataset_name}' (train={max_train:,}, val={max_val:,})...")
    ds = load_dataset(dataset_name, split="train", streaming=True)

    pairs: List[Tuple[str, str]] = []
    total_needed = max_train + max_val

    for item in tqdm(ds, total=total_needed, desc="Loading Query-Code Pairs"):
        query = (item.get("docstring") or item.get("func_documentation_string") or "").strip()
        raw_code = (item.get("code") or item.get("func_code_string") or "").strip()
        if len(query) < 15 or len(raw_code) < 30:
            continue
        code = strip_docstring(raw_code, query)
        if len(code) < 30:
            continue
        pairs.append((query, code))
        if len(pairs) >= total_needed:
            break

    if len(pairs) <= max_val:
        raise RuntimeError(f"Insufficient pairs loaded ({len(pairs)}) for val size {max_val}.")

    val_pairs = pairs[:max_val]
    train_pairs = pairs[max_val:]
    print(f"Loaded {len(train_pairs):,} train pairs and {len(val_pairs):,} held-out val pairs.")
    return train_pairs, val_pairs


def encode_texts(
    tokenizer: Tokenizer,
    texts: List[str],
    max_len: int,
    pad_token_id: int,
    suffix: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch-tokenize texts with optional causal suffix and pad to batch max length."""
    if suffix:
        texts = [t + suffix for t in texts]
    encodings = tokenizer.encode_batch(texts)

    seqs = [enc.ids[:max_len] if enc.ids else [pad_token_id] for enc in encodings]
    batch_max = max(len(s) for s in seqs)

    input_ids = torch.full((len(seqs), batch_max), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), batch_max), dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = len(s)
        input_ids[i, :n] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, :n] = True

    return input_ids, attention_mask


@torch.no_grad()
def evaluate_retrieval(
    model: ShortcakeSPLADE,
    tokenizer: Tokenizer,
    val_pairs: List[Tuple[str, str]],
    max_query_len: int,
    max_doc_len: int,
    pad_token_id: int,
    batch_size: int,
    device: str,
    amp_dtype: torch.dtype,
    use_suffix: bool = True,
) -> Dict[str, float]:
    """Compute real MRR@10, Recall@1, Recall@10, and sparsity (avg active terms) on held-out pairs."""
    model.eval()
    q_vecs = []
    d_vecs = []
    q_suffix = QUERY_SUFFIX if use_suffix else ""
    d_suffix = DOC_SUFFIX if use_suffix else ""

    for i in range(0, len(val_pairs), batch_size):
        batch = val_pairs[i : i + batch_size]
        queries = [p[0] for p in batch]
        docs = [p[1] for p in batch]

        q_ids, q_mask = encode_texts(tokenizer, queries, max_query_len, pad_token_id, suffix=q_suffix)
        d_ids, d_mask = encode_texts(tokenizer, docs, max_doc_len, pad_token_id, suffix=d_suffix)

        q_ids, q_mask = q_ids.to(device), q_mask.to(device)
        d_ids, d_mask = d_ids.to(device), d_mask.to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            q_out = model(q_ids, attention_mask=q_mask)["sparse_vector"]
            d_out = model(d_ids, attention_mask=d_mask)["sparse_vector"]

        q_vecs.append(q_out.float().cpu())
        d_vecs.append(d_out.float().cpu())

    Q = torch.cat(q_vecs, dim=0)  # (N, V)
    D = torch.cat(d_vecs, dim=0)  # (N, V)
    N = Q.size(0)

    # Exact inverted-index dot-product similarity matrix
    scores = torch.matmul(Q, D.T)  # (N, N)
    targets = torch.arange(N)

    # Top-10 rankings per query
    topk_indices = torch.topk(scores, k=min(10, N), dim=-1).indices  # (N, 10)
    matches = topk_indices.eq(targets.unsqueeze(-1))  # (N, 10)

    ranks = torch.arange(1, matches.size(1) + 1, dtype=torch.float32)
    mrr10 = (matches.float() / ranks.unsqueeze(0)).sum(dim=-1).mean().item()
    recall1 = matches[:, 0].float().mean().item()
    recall10 = matches.any(dim=-1).float().mean().item()

    avg_q_terms = (Q > 0).float().sum(dim=-1).mean().item()
    avg_d_terms = (D > 0).float().sum(dim=-1).mean().item()

    model.train()
    return {
        "mrr@10": mrr10,
        "recall@1": recall1,
        "recall@10": recall10,
        "avg_q_terms": avg_q_terms,
        "avg_d_terms": avg_d_terms,
    }


def train_splade(
    base_checkpoint: str = "checkpoints/best_model_44.4m.pt",
    tokenizer_path: str = "tokenizer.json",
    output_dir: str = "checkpoints_splade",
    dataset_name: str = "Nan-Do/code-search-net-python",
    max_train_pairs: int = 50000,
    val_size: int = 1000,
    max_steps: int = 2000,
    batch_size: int = 32,
    max_query_len: int = 128,
    max_doc_len: int = 512,
    learning_rate: float = 5e-5,
    lambda_q: float = 3e-4,
    lambda_d: float = 1e-4,
    flops_warmup_steps: int = 1000,
    temperature: float = 0.05,
    use_suffix: bool = True,
    eval_interval: int = 250,
):
    """Fine-tune ShortcakeSPLADE with symmetric InfoNCE + quadratic FLOPs regularization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        pad_token_id = 1

    print(f"=== Starting Shortcake SPLADE Training ({device}) ===")
    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", ModelConfig())
    backbone = Shortcake(config)
    backbone.load_state_dict(ckpt["model_state_dict"])

    model = ShortcakeSPLADE(backbone, tie_lm_head=True, pad_token_id=pad_token_id).to(device)
    print(f"Loaded backbone ({backbone.get_num_params():,} non-emb params) with tied SPLADE head.")

    train_pairs, val_pairs = load_code_retrieval_pairs(
        dataset_name=dataset_name,
        max_train=max_train_pairs,
        max_val=val_size,
    )

    optimizer = configure_optimizers(model, learning_rate=learning_rate, weight_decay=0.01)
    use_bf16 = (device == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))

    warmup_steps = max(1, min(150, max_steps // 10))
    min_lr = learning_rate * 0.1
    q_suffix = QUERY_SUFFIX if use_suffix else ""
    d_suffix = DOC_SUFFIX if use_suffix else ""

    # Baseline zero-shot evaluation
    base_metrics = evaluate_retrieval(
        model, tokenizer, val_pairs, max_query_len, max_doc_len,
        pad_token_id, batch_size, device, amp_dtype, use_suffix=use_suffix,
    )
    print(
        f"Baseline (Step 0) | MRR@10: {base_metrics['mrr@10']:.4f} | "
        f"R@1: {base_metrics['recall@1']:.4f} | R@10: {base_metrics['recall@10']:.4f} | "
        f"Active Terms (Q/D): {base_metrics['avg_q_terms']:.1f}/{base_metrics['avg_d_terms']:.1f}",
        flush=True,
    )

    best_mrr = base_metrics["mrr@10"]
    step = 0
    num_train = len(train_pairs)
    t0 = time.time()
    model.train()

    while step < max_steps:
        perm = torch.randperm(num_train).tolist()
        for start in range(0, num_train - batch_size + 1, batch_size):
            if step >= max_steps:
                break

            lr = get_lr(step, warmup_steps, max_steps, learning_rate, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Quadratic FLOPs regularization warmup (standard SPLADE schedule)
            reg_scale = min(1.0, (step / max(flops_warmup_steps, 1)) ** 2)
            cur_lq = lambda_q * reg_scale
            cur_ld = lambda_d * reg_scale

            batch = [train_pairs[idx] for idx in perm[start : start + batch_size]]
            q_ids, q_mask = encode_texts(tokenizer, [p[0] for p in batch], max_query_len, pad_token_id, suffix=q_suffix)
            d_ids, d_mask = encode_texts(tokenizer, [p[1] for p in batch], max_doc_len, pad_token_id, suffix=d_suffix)
            q_ids, q_mask = q_ids.to(device), q_mask.to(device)
            d_ids, d_mask = d_ids.to(device), d_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                q_out = model(q_ids, attention_mask=q_mask)
                d_out = model(d_ids, attention_mask=d_mask)

                q_vec = q_out["sparse_vector"].float()
                d_vec = d_out["sparse_vector"].float()

                # Normalize temperature-scaled dot products for stable InfoNCE
                sim = torch.matmul(q_vec, d_vec.T) / max(temperature, 1e-4)
                labels = torch.arange(batch_size, device=device)
                ce_loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))

                flops_q = q_out["flops_loss"]
                flops_d = d_out["flops_loss"]
                loss = ce_loss + cur_lq * flops_q + cur_ld * flops_d

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

            if step % 50 == 0 or step == max_steps:
                q_nnz = (q_vec.detach() > 0).float().sum(dim=-1).mean().item()
                d_nnz = (d_vec.detach() > 0).float().sum(dim=-1).mean().item()
                print(
                    f"Step {step:4d}/{max_steps} | CE: {ce_loss.item():.4f} | "
                    f"FLOPs(Q/D): {flops_q.item():.2f}/{flops_d.item():.2f} | "
                    f"Terms(Q/D): {q_nnz:.1f}/{d_nnz:.1f} | LR: {lr:.6f}",
                    flush=True,
                )

            if step % eval_interval == 0 or step == max_steps:
                metrics = evaluate_retrieval(
                    model, tokenizer, val_pairs, max_query_len, max_doc_len,
                    pad_token_id, batch_size, device, amp_dtype, use_suffix=use_suffix,
                )
                print(
                    f"\n>>> EVAL Step {step}/{max_steps} | MRR@10: {metrics['mrr@10']:.4f} | "
                    f"R@1: {metrics['recall@1']:.4f} | R@10: {metrics['recall@10']:.4f} | "
                    f"Active Terms (Q/D): {metrics['avg_q_terms']:.1f}/{metrics['avg_d_terms']:.1f}\n",
                    flush=True,
                )
                if metrics["mrr@10"] > best_mrr:
                    best_mrr = metrics["mrr@10"]
                    save_path = os.path.join(output_dir, "best_splade_model.pt")
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "metrics": metrics,
                            "use_suffix": use_suffix,
                        },
                        save_path,
                    )
                    print(f" Saved best SPLADE checkpoint -> {save_path} (MRR@10={best_mrr:.4f})", flush=True)

    print(f"\nSPLADE training complete in {(time.time() - t0) / 60:.1f} min! Best MRR@10: {best_mrr:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ShortcakeSPLADE for sparse code/text retrieval.")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/best_model_44.4m.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints_splade")
    parser.add_argument("--dataset", type=str, default="Nan-Do/code-search-net-python")
    parser.add_argument("--max_train_pairs", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_query_len", type=int, default=128)
    parser.add_argument("--max_doc_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda_q", type=float, default=3e-4)
    parser.add_argument("--lambda_d", type=float, default=1e-4)
    parser.add_argument("--flops_warmup_steps", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--no_suffix", action="store_true", help="Disable causal keyword suffix prompt.")
    parser.add_argument("--eval_interval", type=int, default=250)
    args = parser.parse_args()

    train_splade(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        max_train_pairs=args.max_train_pairs,
        val_size=args.val_size,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_query_len=args.max_query_len,
        max_doc_len=args.max_doc_len,
        learning_rate=args.lr,
        lambda_q=args.lambda_q,
        lambda_d=args.lambda_d,
        flops_warmup_steps=args.flops_warmup_steps,
        temperature=args.temperature,
        use_suffix=not args.no_suffix,
        eval_interval=args.eval_interval,
    )


if __name__ == "__main__":
    main()
