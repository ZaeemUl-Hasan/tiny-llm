"""
Step 1: the tokenizer.

A language model never sees text directly. It sees a list of integers ("tokens").
The tokenizer is the translation layer between the two.

We use the simplest possible tokenizer: one token per character. Real LLMs use
subword tokenizers (e.g. BPE), where common chunks like "ing" or " the" get their
own token. That makes sequences shorter, but the idea is exactly the same:
text -> list of ints -> model -> list of ints -> text.
"""

import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string -> int
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # int -> string

    @classmethod
    def from_text(cls, text):
        return cls(text)

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.chars, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))


if __name__ == "__main__":
    tok = CharTokenizer.from_text("hello world")
    ids = tok.encode("hello")
    print("vocab:", tok.chars)
    print("encode('hello') ->", ids)
    print("decode(...)     ->", tok.decode(ids))
