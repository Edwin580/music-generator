"""Model zoo, factory and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..config import ExperimentConfig
from ..data.tokenizer import REMITokenizer
from .lstm import MusicLSTM, PianoRollLSTM
from .transformer import MusicTransformer

__all__ = [
    "MusicLSTM",
    "MusicTransformer",
    "PianoRollLSTM",
    "build_model",
    "count_parameters",
    "load_checkpoint",
    "save_checkpoint",
]


def build_model(cfg: ExperimentConfig, tokenizer: REMITokenizer) -> nn.Module:
    m = cfg.model
    if m.arch == "transformer":
        return MusicTransformer(
            tokenizer.vocab_size, m.d_model, m.n_layers, m.n_heads, m.ff_mult, m.dropout,
            m.max_len, m.tie_embeddings, tokenizer.pad_id,
        )
    if m.arch == "lstm":
        return MusicLSTM(
            tokenizer.vocab_size, m.d_model, m.n_layers, m.dropout, m.tie_embeddings,
            tokenizer.pad_id,
        )
    if m.arch == "pianoroll_lstm":
        return PianoRollLSTM(tokenizer.n_pitches, m.d_model, m.n_layers, m.dropout)
    raise ValueError(f"Unknown architecture {m.arch!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: str | Path, model: nn.Module, cfg: ExperimentConfig,
                    tokenizer: REMITokenizer, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Returns ``(model, config, tokenizer, extra)`` with the model in eval mode."""
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ExperimentConfig.from_dict(payload["config"])
    tokenizer = REMITokenizer.from_dict(payload["tokenizer"])
    model = build_model(cfg, tokenizer)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, cfg, tokenizer, payload.get("extra", {})
