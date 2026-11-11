import os
import torch
import matplotlib.pyplot as plt
from dataset import load_data, get_batch
from model import GPT2
from tokenizer import CharTokenizer, BPETokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/best_model.pt"


def run_tokenizer_comparison(text_sample):
    print("\n" + "=" * 65)
    print("EXPERIMENT: CHARACTER-LEVEL vs. SUBWORD BPE TOKENIZER")
    print("=" * 65)

    test_sentence = "First Citizen:\nBefore we proceed any further, hear me speak."
    
    char_tok = CharTokenizer(text_sample)
    char_encoded = char_tok.encode(test_sentence)
    
    bpe_tok = BPETokenizer(vocab_size=160)
    bpe_tok.train(text_sample[:30000])
    bpe_encoded = bpe_tok.encode(test_sentence)
    
    ratio = len(char_encoded) / len(bpe_encoded)
    
    print(f"Test Sentence: {repr(test_sentence)}")
    print(f"Character-Level Tokens: {len(char_encoded)} (Vocab: {char_tok.vocab_size})")
    print(f"BPE Subword Tokens:     {len(bpe_encoded)} (Vocab: {bpe_tok.vocab_size})")
    print(f"Compression Factor:     {ratio:.2f}x")
    print(f"Learned Subwords:       {[bpe_tok.vocab[t] for t in bpe_encoded[:10]]}")


def run_temperature_experiment(model, tokenizer):
    print("\n" + "=" * 65)
    print("EXPERIMENT 1: EFFECT OF TEMPERATURE")
    print("=" * 65)

    prompt = "ROMEO:\n"
    encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    temperatures = [0.0, 0.2, 0.5, 0.8, 1.2, 1.8]

    for temp in temperatures:
        torch.manual_seed(42)
        gen = model.generate(encoded.clone(), max_new_tokens=100, temperature=temp)
        text = tokenizer.decode(gen[0].tolist())
        label = "Greedy" if temp == 0.0 else f"T = {temp}"
        print(f"\n--- {label} ---")
        print(text.strip())


def run_prompt_context_experiment(model, tokenizer):
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: EFFECT OF PROMPT CONTEXT")
    print("=" * 65)

    prompts = [
        ("Empty / Unconditioned", "\n"),
        ("Short (1 Word)", "KING:"),
        ("Medium (1 Sentence)", "Shall I compare thee to a summer's day?"),
        ("Rich Dialogue Context", "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\n")
    ]

    for name, p in prompts:
        encoded = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=DEVICE)
        torch.manual_seed(42)
        gen = model.generate(encoded, max_new_tokens=100, temperature=0.8)
        text = tokenizer.decode(gen[0].tolist())
        print(f"\n--- Context: {name} ({len(p)} chars) ---")
        print(text.strip())


def run_ablation_study(train_data, val_data, tokenizer):
    print("\n" + "=" * 65)
    print("EXPERIMENT 4: ARCHITECTURAL ABLATION (4-LAYER vs 2-LAYER)")
    print("=" * 65)

    configs = [
        ("4-Layer Baseline", 4),
        ("2-Layer Ablation", 2),
    ]

    history = {}
    steps = 400

    for name, n_layer in configs:
        print(f"\n[*] Training {name} ({n_layer} layers)...")
        torch.manual_seed(42)
        m = GPT2(
            vocab_size=tokenizer.vocab_size,
            block_size=128,
            n_layer=n_layer,
            n_head=4,
            d_model=128
        ).to(DEVICE)

        opt = torch.optim.AdamW(m.parameters(), lr=6e-4)
        history[name] = []

        for s in range(1, steps + 1):
            xb, yb = get_batch(train_data, 64, 128, DEVICE)
            _, loss = m(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if s % 100 == 0 or s == steps:
                m.eval()
                val_losses = []
                for _ in range(20):
                    vx, vy = get_batch(val_data, 64, 128, DEVICE)
                    with torch.no_grad():
                        _, vloss = m(vx, vy)
                    val_losses.append(vloss.item())
                m.train()
                avg_val = sum(val_losses) / len(val_losses)
                history[name].append((s, avg_val))
                print(f"    Step {s:3d}/{steps} | Validation Loss: {avg_val:.4f}")

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for name, points in history.items():
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        plt.plot(x_vals, y_vals, label=name, marker="o", linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel("Validation Loss")
    plt.title("Ablation Study: 4 Layers vs 2 Layers")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/ablation_depth.png", dpi=300)
    plt.close()
    print("\n[+] Saved plot to results/ablation_depth.png")


def main():
    train_data, val_data, tokenizer = load_data()

    with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    run_tokenizer_comparison(full_text)

    if os.path.exists(CHECKPOINT_PATH):
        cp = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model = GPT2(
            vocab_size=cp["vocab_size"],
            block_size=cp["block_size"],
            n_layer=cp["n_layer"],
            n_head=cp["n_head"],
            d_model=cp["d_model"]
        ).to(DEVICE)
        model.load_state_dict(cp["model_state_dict"])
        model.eval()

        run_temperature_experiment(model, tokenizer)
        run_prompt_context_experiment(model, tokenizer)
    else:
        print(f"[*] Checkpoint '{CHECKPOINT_PATH}' not found. Run train.py first to test generation.")

    run_ablation_study(train_data, val_data, tokenizer)


if __name__ == "__main__":
    main()
