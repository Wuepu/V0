
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8000
    seq_len: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        b, t, c = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                b, t, self.n_head, self.head_dim
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(b, t, c)

        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attention = CausalAttention(cfg)

        self.ln2 = nn.LayerNorm(cfg.d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        assert cfg.d_model % cfg.n_head == 0

        self.cfg = cfg

        self.token_embedding = nn.Embedding(
            cfg.vocab_size, cfg.d_model
        )

        self.position_embedding = nn.Embedding(
            cfg.seq_len, cfg.d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg)
            for _ in range(cfg.n_layer)
        ])

        self.final_norm = nn.LayerNorm(cfg.d_model)

        self.output = nn.Linear(
            cfg.d_model, cfg.vocab_size, bias=False
        )

        self.apply(self._initialize_weights)

        self.output.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        batch_size, seq_len = tokens.shape

        assert seq_len <= self.cfg.seq_len

        positions = torch.arange(
            seq_len, device=tokens.device
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)[None, :, :]
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.output(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss
