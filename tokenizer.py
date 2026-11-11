class CharTokenizer:
    """Character-level tokenizer."""
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, token_ids: list[int]) -> str:
        return "".join([self.itos[i] for i in token_ids if i in self.itos])


class BPETokenizer:
    """Byte-Pair Encoding (BPE) subword tokenizer."""
    def __init__(self, vocab_size: int = 256):
        self.target_vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.stoi = {}
        self.vocab_size = 0

    @staticmethod
    def _get_stats(tokens: list[int]) -> dict:
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(tokens: list[int], pair: tuple, new_id: int) -> list[int]:
        new_tokens = []
        i = 0
        n = len(tokens)
        p0, p1 = pair
        while i < n:
            if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        tokens = [self.stoi[c] for c in text]

        num_merges = self.target_vocab_size - len(chars)
        for _ in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = len(self.vocab)
            tokens = self._merge(tokens, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list[int]:
        tokens = [self.stoi[c] for c in text if c in self.stoi]
        while len(tokens) >= 2:
            pairs = list(zip(tokens, tokens[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = self._merge(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        return "".join([self.vocab[i] for i in token_ids if i in self.vocab])


if __name__ == "__main__":
    sample = "First Citizen:\nBefore we proceed any further, hear me speak."
    tok = CharTokenizer(sample)
    print("Vocab:", tok.vocab_size, "| Encoded:", tok.encode("Hello"))
