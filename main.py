"""
Master Execution Pipeline for GPT-2 From Scratch.

Runs the complete assignment workflow end-to-end:
1. Tokenizer comparison (Character-Level vs. Subword BPE)
2. Baseline model training with validation tracking & checkpointing
3. Text generation experiments (Temperature sweep & Prompt context)
4. Loss dynamics visualization (Saved to results/loss_dynamics.png)
5. Architectural ablation study (4 Layers vs. 2 Layers, saved to results/ablation_depth.png)

Usage:
    python main.py
"""

import math
import os
import torch
import matplotlib.pyplot as plt

from dataset import load_data, get_batch
from model import GPT2
from tokenizer import CharTokenizer, BPETokenizer

# Hyperparameters (calibrated for fast, reliable CPU or GPU execution)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
BLOCK_SIZE = 64
LEARNING_RATE = 8e-4
DROPOUT = 0.1

# Baseline Architecture
D_MODEL = 64
N_HEAD = 4
N_LAYER = 4

# Training Steps
MAX_ITERS = 600
EVAL_INTERVAL = 100
EVAL_ITERS = 25
ABLATION_STEPS = 300


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
            _, loss = model(x, y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        out[split] = mean_loss
        out[f"{split}_ppl"] = math.exp(mean_loss)
    model.train()
    return out


def stage_1_tokenizers(raw_text):
    print("\n" + "=" * 70)
    print("  [STAGE 1/5] TOKENIZATION: CHARACTER-LEVEL vs. SUBWORD BPE")
    print("=" * 70)

    test_sentence = "First Citizen:\nBefore we proceed any further, hear me speak."
    
    char_tok = CharTokenizer(raw_text)
    char_ids = char_tok.encode(test_sentence)

    bpe_tok = BPETokenizer(vocab_size=160)
    bpe_tok.train(raw_text[:30000])
    bpe_ids = bpe_tok.encode(test_sentence)

    ratio = len(char_ids) / len(bpe_ids)
    print(f"Test Sentence: {repr(test_sentence)}")
    print(f"Character-Level Tokens: {len(char_ids)} (Vocab: {char_tok.vocab_size})")
    print(f"BPE Subword Tokens:     {len(bpe_ids)} (Vocab: {bpe_tok.vocab_size})")
    print(f"Compression Factor:     {ratio:.2f}x fewer tokens with BPE")
    print(f"Learned Subwords:       {[bpe_tok.vocab[t] for t in bpe_ids[:10]]}")
    return char_tok


def stage_2_train_baseline(train_data, val_data, tokenizer):
    print("\n" + "=" * 70)
    print(f"  [STAGE 2/5] TRAINING BASELINE GPT-2 (4 Layers, 4 Heads, d={D_MODEL})")
    print("=" * 70)

    model = GPT2(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        d_model=D_MODEL,
        dropout=DROPOUT
    ).to(DEVICE)
    print(f"[*] Total trainable parameters: {model.get_num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/best_model.pt"

    history = {"step": [], "train_loss": [], "val_loss": [], "val_ppl": []}
    best_val_loss = float("inf")

    for step in range(1, MAX_ITERS + 1):
        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS:
            losses = estimate_loss(model, train_data, val_data)
            history["step"].append(step)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            history["val_ppl"].append(losses["val_ppl"])

            print(f"Step {step:4d}/{MAX_ITERS} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Val PPL: {losses['val_ppl']:.2f}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "vocab_size": tokenizer.vocab_size,
                    "block_size": BLOCK_SIZE,
                    "n_layer": N_LAYER,
                    "n_head": N_HEAD,
                    "d_model": D_MODEL,
                    "val_loss": best_val_loss,
                    "chars": tokenizer.chars
                }, checkpoint_path)

    print(f"[+] Baseline training complete. Best Validation Loss: {best_val_loss:.4f}")
    return model, history


def stage_3_generation_experiments(model, tokenizer):
    print("\n" + "=" * 70)
    print("  [STAGE 3/5] TEXT GENERATION EXPERIMENTS")
    print("=" * 70)

    # 1. Temperature Experiment
    print("\n--- Experiment 1: Effect of Temperature ---")
    prompt = "ROMEO:\n"
    encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    temperatures = [0.0, 0.2, 0.5, 0.8, 1.2, 1.8]

    for temp in temperatures:
        torch.manual_seed(42)
        gen = model.generate(encoded.clone(), max_new_tokens=80, temperature=temp)
        label = "Greedy (T=0.0)" if temp == 0.0 else f"T = {temp}"
        sample = tokenizer.decode(gen[0].tolist()).strip().replace("\n", " ")
        print(f"[{label:12s}] {sample[:90]}...")

    # 2. Prompt Context Experiment
    print("\n--- Experiment 2: Effect of Prompt Context (T=0.8) ---")
    prompts = [
        ("Empty", "\n"),
        ("Short", "KING:"),
        ("Sentence", "Shall I compare thee to a summer's day?"),
        ("Dialogue", "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\n")
    ]

    for name, p in prompts:
        enc = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=DEVICE)
        torch.manual_seed(42)
        gen = model.generate(enc, max_new_tokens=80, temperature=0.8)
        sample = tokenizer.decode(gen[0].tolist()).strip().replace("\n", " ")
        print(f"[{name:9s}] {sample[:90]}...")


def stage_4_plot_loss_dynamics(history):
    print("\n" + "=" * 70)
    print("  [STAGE 4/5] PLOTTING LOSS DYNAMICS & PERPLEXITY")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(history["step"], history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["step"], history["val_loss"], label="Val Loss", linestyle="--", linewidth=2)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(history["step"], history["val_ppl"], label="Validation Perplexity", color="green", linewidth=2)
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Perplexity (exp(Loss))")
    ax2.set_title("Validation Perplexity Trajectory")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    out_path = "results/loss_dynamics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved loss dynamics plot -> {out_path}")


def stage_5_ablation_study(train_data, val_data, tokenizer, baseline_history):
    print("\n" + "=" * 70)
    print("  [STAGE 5/5] ARCHITECTURAL ABLATION: 4 LAYERS vs. 2 LAYERS")
    print("=" * 70)

    print(f"[*] Training ablated 2-layer model for {ABLATION_STEPS} steps...")
    torch.manual_seed(42)
    ablated_model = GPT2(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=2,
        n_head=N_HEAD,
        d_model=D_MODEL,
        dropout=DROPOUT
    ).to(DEVICE)
    print(f"[*] Ablated model parameters: {ablated_model.get_num_params():,}")

    optimizer = torch.optim.AdamW(ablated_model.parameters(), lr=LEARNING_RATE)
    ablation_history = {"step": [], "val_loss": []}

    for step in range(1, ABLATION_STEPS + 1):
        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        _, loss = ablated_model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == ABLATION_STEPS:
            m = estimate_loss(ablated_model, train_data, val_data)
            ablation_history["step"].append(step)
            ablation_history["val_loss"].append(m["val"])
            print(f"Ablation Step {step:4d}/{ABLATION_STEPS} | Val Loss: {m['val']:.4f}")

    # Comparative plot
    plt.figure(figsize=(8, 4.5))
    base_steps = [s for s in baseline_history["step"] if s <= ABLATION_STEPS]
    base_losses = baseline_history["val_loss"][:len(base_steps)]

    plt.plot(base_steps, base_losses, label=f"4-Layer Baseline ({D_MODEL}d)", marker="o", linewidth=2)
    plt.plot(ablation_history["step"], ablation_history["val_loss"], label=f"2-Layer Ablated ({D_MODEL}d)", marker="s", linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel("Validation Loss")
    plt.title("Ablation Study: Impact of Model Depth on Convergence")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_path = "results/ablation_depth.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved ablation comparison plot -> {out_path}")


def main():
    torch.manual_seed(42)
    print("=" * 70)
    print(f"  GPT-2 FROM SCRATCH: FULL ASSIGNMENT EXECUTION PIPELINE")
    print(f"  Device: {DEVICE.upper()}")
    print("=" * 70)

    train_data, val_data, tokenizer = load_data()
    with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    # Stage 1: Tokenizer Comparison (Char vs BPE)
    stage_1_tokenizers(full_text)

    # Stage 2: Train Baseline Model
    model, baseline_history = stage_2_train_baseline(train_data, val_data, tokenizer)

    # Stage 3: Generation Experiments (Temperature & Context)
    stage_3_generation_experiments(model, tokenizer)

    # Stage 4: Plot Loss Dynamics
    stage_4_plot_loss_dynamics(baseline_history)

    # Stage 5: Architectural Ablation Study
    stage_5_ablation_study(train_data, val_data, tokenizer, baseline_history)

    print("\n" + "=" * 70)
    print("  ALL 5 STAGES COMPLETED SUCCESSFULLY!")
    print("  Artifacts generated:")
    print("   • Model Checkpoint: checkpoints/best_model.pt")
    print("   • Loss Curves:     results/loss_dynamics.png")
    print("   • Ablation Plot:   results/ablation_depth.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
