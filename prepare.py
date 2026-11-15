"""
Step 0: get some training data.

Downloads "Tiny Shakespeare" (about 1 MB of Shakespeare's plays, public domain)
and builds the character tokenizer from it.

Swap in any .txt file you like: song lyrics, your own notes, code. The model will
learn to write in whatever style it is trained on.
"""

import os
import urllib.request

from tokenizer import CharTokenizer

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data"
TEXT_PATH = os.path.join(DATA_DIR, "input.txt")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(TEXT_PATH):
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, TEXT_PATH)

    with open(TEXT_PATH, encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer.from_text(text)
    tok.save(TOKENIZER_PATH)

    print(f"{len(text):,} characters")
    print(f"vocab size: {tok.vocab_size}")
    print(f"vocab: {''.join(tok.chars)!r}")
    print(f"saved tokenizer to {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
