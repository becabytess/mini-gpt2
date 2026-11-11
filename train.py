import math
import os
import torch
from dataset import load_data, get_batch
from model import GPT2

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
BLOCK_SIZE = 64
MAX_ITERS = 600
EVAL_INTERVAL = 100
EVAL_ITERS = 25
LEARNING_RATE = 8e-4

D_MODEL = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1


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


def main():
    torch.manual_seed(42)
    print(f"[*] Training on: {DEVICE}")

    train_data, val_data, tokenizer = load_data()
    print(f"[*] Tokens: Train={len(train_data):,} | Val={len(val_data):,} | Vocab={tokenizer.vocab_size}")

    model = GPT2(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        d_model=D_MODEL,
        dropout=DROPOUT
    ).to(DEVICE)
    print(f"[*] Model parameters: {model.get_num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/best_model.pt"

    for step in range(1, MAX_ITERS + 1):
        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS:
            losses = estimate_loss(model, train_data, val_data)
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

        if step % 200 == 0:
            prompt = "ROMEO:\n"
            p_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
            gen = model.generate(p_tokens, max_new_tokens=60, temperature=0.8)
            print("-" * 50)
            print(f"Preview (Step {step}):\n{tokenizer.decode(gen[0].tolist())}")
            print("-" * 50)

    print(f"[+] Done. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
