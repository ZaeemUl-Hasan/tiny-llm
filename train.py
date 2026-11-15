"""
Step 3: training.

The training loop, in plain words:
  1. Grab random chunks of text. The input is the chunk; the target is the same
     chunk shifted one character to the left (the "next character" at every position).
  2. Run the model, measure how wrong its predictions were (the loss).
  3. Backpropagation works out how each parameter should change to reduce the loss.
  4. The optimizer nudges every parameter a tiny bit in that direction.
  5. Repeat a few thousand times.

Run:  python train.py            (full run, about 45 minutes on an M4 Mac)
      python train.py --quick    (tiny model, about a minute, to check everything works)
"""

import argparse
import os
import time

import torch

from model import GPT, GPTConfig
from prepare import TEXT_PATH, TOKENIZER_PATH
from tokenizer import CharTokenizer

OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny model for a fast smoke test")
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()

    # hyperparameters: the knobs you set before training. Try changing them!
    if args.quick:
        config = GPTConfig(block_size=64, n_layer=2, n_head=4, n_embd=128, dropout=0.0)
        batch_size, max_iters, eval_interval, learning_rate = 32, 500, 100, 1e-3
    else:
        config = GPTConfig(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2)
        # val loss stops improving around step 3000; training longer only overfits
        batch_size, max_iters, eval_interval, learning_rate = 64, 3000, 250, 3e-4
    if args.max_iters is not None:
        max_iters = args.max_iters
    eval_iters = 50

    torch.manual_seed(1337)
    device = get_device()
    print(f"using device: {device}")

    if not os.path.exists(TOKENIZER_PATH):
        raise SystemExit("no data yet: run `python prepare.py` first")
    tok = CharTokenizer.load(TOKENIZER_PATH)
    with open(TEXT_PATH, encoding="utf-8") as f:
        data = torch.tensor(tok.encode(f.read()), dtype=torch.long)

    # hold back 10% of the text the model never trains on. If the loss on this
    # "validation" text stops improving while training loss keeps falling, the model
    # is memorizing instead of learning (overfitting).
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    def get_batch(split):
        d = splits[split]
        ix = torch.randint(len(d) - config.block_size - 1, (batch_size,))
        x = torch.stack([d[i : i + config.block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + config.block_size + 1] for i in ix])  # shifted by one
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()  # turns off dropout
        out = {}
        for split in splits:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    config.vocab_size = tok.vocab_size
    model = GPT(config).to(device)
    print(f"model has {model.num_params() / 1e6:.2f}M parameters")
    # a model that guesses uniformly at random would score ln(vocab_size)
    print(f"loss of random guessing: {torch.log(torch.tensor(float(tok.vocab_size))).item():.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    os.makedirs(OUT_DIR, exist_ok=True)
    start = time.time()
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {time.time() - start:.0f}s")

        if it == max_iters:
            break

        x, y = get_batch("train")
        _, loss = model(x, y)              # forward pass: make predictions, measure error
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                    # backpropagation: compute gradients
        optimizer.step()                   # update the parameters

    torch.save({"model": model.state_dict(), "config": config.__dict__}, CKPT_PATH)
    print(f"saved checkpoint to {CKPT_PATH}")

    model.eval()
    prompt = torch.tensor([tok.encode("\n")], device=device)
    print("\n--- sample ---")
    print(tok.decode(model.generate(prompt, 300, temperature=0.8)[0].tolist()))


if __name__ == "__main__":
    main()
