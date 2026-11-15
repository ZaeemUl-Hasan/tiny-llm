"""
Step 4: generate text from a trained model.

Run:  python sample.py
      python sample.py --prompt "ROMEO:" --tokens 500 --temperature 0.8
"""

import argparse

import torch

from model import GPT, GPTConfig
from prepare import TOKENIZER_PATH
from tokenizer import CharTokenizer
from train import CKPT_PATH, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    device = get_device()
    tok = CharTokenizer.load(TOKENIZER_PATH)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    idx = torch.tensor([tok.encode(args.prompt)], device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
