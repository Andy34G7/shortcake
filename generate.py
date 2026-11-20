import os
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import ModelConfig
from model import Shortcake


def sample_top_p_top_k(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> torch.Tensor:
    """Filter logits using repetition penalty, temperature (or greedy), top-k, and top-p."""
    logits = logits.clone()

    # Vectorized repetition penalty (no GPU->CPU sync)
    if repetition_penalty != 1.0 and input_ids.numel() > 0:
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(1, input_ids, score)

    # Greedy decoding when temperature <= 0
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # Top-K filtering
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

    # Top-P (Nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


@torch.no_grad()
def generate_code(
    model: Shortcake,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: str = "cpu",
) -> str:
    """Generate text completion using SSM state + Attention KV caching."""
    model.eval()
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
    eos_token_id = tokenizer.token_to_id("<eos>")

    # Prefill prompt once to populate SSM states & KV cache
    cache = model.init_cache()
    cond_ids = input_ids[:, -model.config.max_seq_len :]
    outputs = model(cond_ids, cache=cache)
    next_token_logits = outputs["logits"][:, -1, :]

    for _ in range(max_new_tokens):
        next_token = sample_top_p_top_k(
            next_token_logits,
            input_ids=input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        input_ids = torch.cat((input_ids, next_token), dim=1)
        # Single-token O(1) Mamba + cached KV Attention decode step
        outputs = model(next_token, cache=cache)
        next_token_logits = outputs["logits"][:, -1, :]

    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate code completions from trained Shortcake checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint.pt file.")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json.")
    parser.add_argument("--prompt", type=str, default="def binary_search(", help="Prompt text.")
    parser.add_argument("--max_tokens", type=int, default=120, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.15, help="Repetition penalty.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Train the model first!")

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", ModelConfig())

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = Shortcake(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"\n--- Code Generation ---")
    print(f"Prompt: {args.prompt}")
    print("------------------------")
    output_text = generate_code(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )
    print(output_text)


if __name__ == "__main__":
    main()
