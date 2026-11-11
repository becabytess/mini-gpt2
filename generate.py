import os
import torch
from model import GPT2
from tokenizer import CharTokenizer

CHECKPOINT_PATH = "checkpoints/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at '{CHECKPOINT_PATH}'. Please run train.py first.")

    cp = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    tokenizer = CharTokenizer("".join(cp["chars"]))
    
    model = GPT2(
        vocab_size=cp["vocab_size"],
        block_size=cp["block_size"],
        n_layer=cp["n_layer"],
        n_head=cp["n_head"],
        d_model=cp["d_model"]
    ).to(DEVICE)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    return model, tokenizer


def generate(prompt="ROMEO:\n", max_tokens=150, temperature=0.8):
    model, tokenizer = load_model()
    encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    out = model.generate(encoded, max_new_tokens=max_tokens, temperature=temperature)
    return tokenizer.decode(out[0].tolist())


if __name__ == "__main__":
    prompt = "ROMEO:\n"
    print(f"Prompt: {repr(prompt)}")
    print("=" * 60)
    print(generate(prompt=prompt, max_tokens=150, temperature=0.8))
    print("=" * 60)
