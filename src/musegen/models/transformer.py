"""Decoder-only Transformer for REMI tokens.

Pre-LayerNorm blocks, rotary position embeddings (RoPE), fused causal attention via
``scaled_dot_product_attention`` and a key/value cache so generation costs O(n) per
token instead of re-running the whole context (the notebook re-ran ``model.predict``
over the full window for every step).

Because RoPE encodes *relative* offsets, the KV cache can be trimmed to a sliding
window of ``max_len`` positions, so generation can run past the training context.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos.to(x.dtype) + _rotate_half(x) * sin.to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        max_cache: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q, k, v = (z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v))
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k[:, :, -max_cache:], v[:, :, -max_cache:])

        # Queries are the last t positions of the key sequence; build the causal mask
        # explicitly when a cache is present (is_causal assumes square attention).
        if past is None:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            s = k.shape[2]
            mask = torch.ones(t, s, dtype=torch.bool, device=x.device).tril(diagonal=s - t)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.proj(out)), present


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin, past, max_cache):
        attn_out, present = self.attn(self.ln1(x), cos, sin, past, max_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class MusicTransformer(nn.Module):
    """Autoregressive Transformer language model over music tokens."""

    kind = "tokens"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 1024,
        tie_embeddings: bool = True,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(d_model // n_heads)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.embed.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():  # GPT-2 style scaled residual init
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(param, std=0.02 / math.sqrt(2 * n_layers))
        with torch.no_grad():  # re-zero the padding row after init
            self.embed.weight[pad_id].zero_()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self, tokens: torch.Tensor, state: dict | None = None
    ) -> tuple[torch.Tensor, dict]:
        """``tokens``: (B, T) ids. ``state`` carries the KV cache and absolute position."""
        past: KVCache | None = state["cache"] if state else None
        offset = state["pos"] if state else 0
        t = tokens.shape[1]
        positions = torch.arange(offset, offset + t, device=tokens.device)
        cos, sin = self.rotary(positions)

        x = self.drop(self.embed(tokens))
        presents: KVCache = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, cos, sin, past[i] if past else None, self.max_len)
            presents.append(present)
        logits = self.head(self.ln_f(x))
        return logits, {"cache": presents, "pos": offset + t}
