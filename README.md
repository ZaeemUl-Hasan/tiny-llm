# tiny-llm

A small GPT-style language model built from scratch in PyTorch, as a way to learn how LLMs actually work.

It trains on Shakespeare and learns to write text that looks like it. It uses the same architecture as ChatGPT-style models (a decoder-only transformer), only about 10,000x smaller.

## How it works

| Step | File | What it does |
|------|------|--------------|
| 0 | [`prepare.py`](prepare.py) | Downloads the training text (Tiny Shakespeare, ~1 MB) |
| 1 | [`tokenizer.py`](tokenizer.py) | Turns text into numbers and back (one token per character) |
| 2 | [`model.py`](model.py) | The transformer: embeddings, self-attention, MLP blocks |
| 3 | [`train.py`](train.py) | The training loop: predict the next character, measure the error, adjust weights |
| 4 | [`sample.py`](sample.py) | Generates new text from the trained model |

Every file is commented to explain the *why*, not just the *what*. Reading them in order is the best way to follow along.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
python prepare.py           # download data, build tokenizer
python train.py --quick     # ~10 seconds: tiny model, checks everything works
python train.py             # full run: ~10.8M parameters, roughly 10-20 min on an Apple M-series Mac
python sample.py --prompt "ROMEO:" --tokens 500
```

Training uses the Apple Silicon GPU (`mps`) or an NVIDIA GPU (`cuda`) when one is available, and falls back to the CPU otherwise.

## Reading the training output

- **loss** measures how surprised the model is by the real next character. Lower is better.
- A model that guesses at random scores about **4.17** (that's ln 65, for 65 possible characters).
- The quick run reaches about **2.0**: it has learned words, spacing and the play format, but spells badly.
- The full run should reach about **1.5**: real-looking words and character names in Shakespearean style.
- If **val loss** starts rising while **train loss** keeps falling, the model is memorizing the training text instead of learning patterns (overfitting).

## Ideas to try next

- Change the hyperparameters in `train.py` (layers, heads, embedding size, learning rate) and compare losses
- Train on your own text: replace `data/input.txt` with any `.txt` file and re-run `prepare.py`
- Replace the character tokenizer with a BPE (subword) tokenizer, like real LLMs use
- Add a learning-rate schedule (warmup, then cosine decay)
- Fine-tune: train on one text, then continue training on another
