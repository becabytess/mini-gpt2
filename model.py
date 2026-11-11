import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer Normalization: y = ((x - mean) / sqrt(var + eps)) * gamma + beta"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # (B, T, d_model)

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, n_head, T, T)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att_weights = self.attn_dropout(F.softmax(att, dim=-1))

        y = att_weights @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, d_model=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_final = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("fc2.weight") or pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))  # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = GPT2(vocab_size=65, block_size=32, n_layer=2, n_head=2, d_model=64)
    x = torch.randint(0, 65, (2, 16))
    logits, loss = m(x, x)
    print("Params:", m.get_num_params(), "| Logits:", logits.shape, "| Loss:", loss.item())
