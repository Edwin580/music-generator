"""Recurrent models.

``MusicLSTM`` is the notebook's Elman-style recurrent idea (Lab 5: *Finding Structure
in Time*) applied to REMI tokens. ``PianoRollLSTM`` is the notebook's original model,
kept for comparison and fixed:

* it outputs **logits** and is trained with ``BCEWithLogitsLoss`` (numerically stable)
  rather than sigmoid + binary cross-entropy;
* it predicts onsets as well as sustains, so repeated notes can be told apart from held ones;
* it carries its recurrent state between steps during generation, so it is not limited
  to a 48-frame window.
"""

from __future__ import annotations

import torch
from torch import nn


class MusicLSTM(nn.Module):
    kind = "tokens"

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3,
        tie_embeddings: bool = True,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        # Scale embeddings so tied output logits start near N(0, 1). With PyTorch's default
        # N(0, 1) init, LayerNorm(h) @ E.T has std ~sqrt(d_model), so the first loss is ~40.
        nn.init.normal_(self.embed.weight, std=d_model**-0.5)
        if tie_embeddings:
            self.head.weight = self.embed.weight
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():  # re-zero the padding row after init
            self.embed.weight[pad_id].zero_()

    def forward(self, tokens: torch.Tensor, state: dict | None = None):
        hidden = state["hidden"] if state else None
        x = self.drop(self.embed(tokens))
        out, hidden = self.lstm(x, hidden)
        logits = self.head(self.ln(self.drop(out)))
        return logits, {"hidden": hidden}


class PianoRollLSTM(nn.Module):
    """Multi-label next-frame predictor over (sustain, onset) piano-roll frames."""

    kind = "pianoroll"

    def __init__(self, n_pitches: int, d_model: int = 256, n_layers: int = 2,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.n_pitches = n_pitches
        self.inp = nn.Linear(2 * n_pitches, d_model)
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 2 * n_pitches)
        # Notes are sparse (~1-2% of cells active); start with a matching prior so
        # early training isn't spent learning "everything is off".
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, frames: torch.Tensor, state: dict | None = None):
        hidden = state["hidden"] if state else None
        x = torch.relu(self.inp(frames))
        out, hidden = self.lstm(x, hidden)
        return self.head(self.drop(out)), {"hidden": hidden}
