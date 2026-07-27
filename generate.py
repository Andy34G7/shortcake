import os
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import ModelConfig
from model import Shortcake


def sample_top_p_top_k(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> torch.Tensor:
    """Filter logits using temperature, top-k, and nucleus (top-p) sampling."""
    logits = logits / max(temperature, 1e-5)

    # Top-K filtering
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

    # Top-P (Nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p threshold
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
    device: str = "cpu",
) -> str:
    """Generate text completion for a given prompt string."""
    model.eval()
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Truncate to max_seq_len if necessary
        cond_ids = input_ids[:, -model.config.max_seq_len :]
        outputs = model(cond_ids)
        next_token_logits = outputs["logits"][:, -1, :]
        next_token = sample_top_p_top_k(next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p)
        input_ids = torch.cat((input_ids, next_token), dim=1)

    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate code completions from trained Shortcake checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint.pt file.")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json.")
    parser.add_argument("--prompt", type=str, default="def binary_search(", help="Prompt text.")
    parser.add_argument("--max_tokens", type=int, default=80, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Train the model first!")

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
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
        device=device,
    )
    print(output_text)


if __name__ == "__main__":
    main()
