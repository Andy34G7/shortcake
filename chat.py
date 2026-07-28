import os
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import ModelConfig
from model import Shortcake
from generate import sample_top_p_top_k


CHATML_SYSTEM_PROMPT = "<|im_start|>system\nYou are Shortcake, a helpful AI assistant.<|im_end|>\n"


@torch.no_grad()
def generate_chat_response(
    model: Shortcake,
    tokenizer: Tokenizer,
    conversation_history: list,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: str = "cpu",
) -> str:
    """Generates an assistant response for an ongoing ChatML conversation."""
    model.eval()

    # Build ChatML prompt history
    prompt = CHATML_SYSTEM_PROMPT
    for msg in conversation_history:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    eos_token_id = tokenizer.token_to_id("<|im_end|>")
    generated_tokens = []

    for _ in range(max_new_tokens):
        cond_ids = input_ids[:, -model.config.max_seq_len :]
        outputs = model(cond_ids)
        next_token_logits = outputs["logits"][:, -1, :]
        next_token = sample_top_p_top_k(
            next_token_logits,
            input_ids=input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        token_item = next_token.item()
        if token_item == eos_token_id:
            break

        input_ids = torch.cat((input_ids, next_token), dim=1)
        generated_tokens.append(token_item)

    response_text = tokenizer.decode(generated_tokens).strip()
    # Clean up residual stop tags if generated
    if response_text.endswith("<|im_end|>"):
        response_text = response_text[:-10].strip()

    return response_text


def main():
    parser = argparse.ArgumentParser(description="Interactive Chat CLI with Shortcake model.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer}")

    print(f"=== Shortcake Interactive Chat CLI ===")
    print(f"Loading checkpoint from {args.checkpoint} on {device}...")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", ModelConfig())
    model = Shortcake(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    tokenizer = Tokenizer.from_file(args.tokenizer)

    conversation_history = []

    print("\nShortcake Assistant is ready! (Type 'exit' or 'clear' to quit/reset)\n" + "-" * 50)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                conversation_history.clear()
                print("Conversation history cleared.\n")
                continue

            conversation_history.append({"role": "user", "content": user_input})
            response = generate_chat_response(
                model=model,
                tokenizer=tokenizer,
                conversation_history=conversation_history,
                temperature=args.temperature,
                device=device,
            )
            print(f"\nShortcake: {response}\n" + "-" * 50)
            conversation_history.append({"role": "assistant", "content": response})

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break


if __name__ == "__main__":
    main()
