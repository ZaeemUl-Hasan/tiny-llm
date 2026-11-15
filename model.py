"""
Step 2: the model, a small GPT-style transformer.

The whole job of the model: given a sequence of tokens, predict the next one.
Everything below is in service of that single goal.

Shapes used in comments:
    B = batch size (how many sequences at once)
    T = sequence length (how many tokens per sequence)
    C = embedding size (how many numbers represent each token)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256  # max context length: how many past tokens the model can see
    n_layer: int = 6       # number of transformer blocks stacked on top of each other
    n_head: int = 6        # number of attention heads per block
    n_embd: int = 384      # embedding size (C)
    dropout: float = 0.2


class CausalSelfAttention(nn.Module):
    """
    Attention lets each token look at earlier tokens and pull in information from them.

    Each token produces three vectors:
      - query: "what am I looking for?"
      - key:   "what do I contain?"
      - value: "what will I hand over if you pick me?"
    A token's query is compared against every earlier token's key. High scores mean
    "this one is relevant", and the output is a weighted mix of those tokens' values.

    "Causal" means a token can only look backwards. Otherwise, during training it
    could just peek at the answer (the next token).
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # one linear layer produces query, key and value for all heads at once
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # lower-triangular mask: position t may only attend to positions <= t
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # split C into n_head smaller heads: (B, T, C) -> (B, n_head, T, head_dim)
        # each head can learn to look for a different kind of relationship
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # attention scores: how much each token cares about each other token
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # hide the future
        att = F.softmax(att, dim=-1)  # turn scores into weights that sum to 1
        att = self.attn_dropout(att)

        y = att @ v  # weighted mix of values: (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # glue the heads back together
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """
    After attention gathers information from other tokens, the MLP lets each token
    "think" about what it gathered, on its own. Most of a model's parameters, and
    much of its stored knowledge, live in these layers.
    """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    One transformer block = attention (communicate) + MLP (compute).

    The "x = x + ..." lines are residual connections: each block adds a small update
    to the running representation instead of replacing it. This is what makes deep
    stacks of blocks trainable. LayerNorm keeps the numbers in a stable range.
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # token embedding: each token id gets its own learned vector of n_embd numbers
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding: attention by itself has no sense of order, so we add
        # a learned vector for "position 0", "position 1", ...
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        # final layer: turn each token's vector into a score for every possible next token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence is longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross-entropy: how surprised was the model by the real next token?
            # lower = better predictions. This one number is what training minimizes.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Text generation is just next-token prediction in a loop:
        predict a probability for every token, pick one, append it, repeat.

        temperature < 1 makes the model more confident and repetitive; > 1 more random.
        top_k only allows the k most likely tokens to be picked.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # only the last block_size tokens fit
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # we only need the prediction for the last position
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
