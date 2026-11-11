import os
import urllib.request
import torch
from tokenizer import CharTokenizer

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "tinyshakespeare.txt"


def load_data():
    if not os.path.exists(DATA_FILE):
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, tokenizer


def get_batch(data, batch_size, block_size, device="cpu"):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


if __name__ == "__main__":
    train_data, val_data, tok = load_data()
    x, y = get_batch(train_data, batch_size=4, block_size=8)
    print("x[0]:", tok.decode(x[0].tolist()), "| y[0]:", tok.decode(y[0].tolist()))
