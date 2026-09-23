"""REMI-style event tokenization of quantized music.

A piece is written as a flat sequence of events (Huang & Yang, 2020, "Pop Music
Transformer")::

    BOS  BAR  POS_0 VEL_5 PITCH_67 DUR_4  POS_4 VEL_5 PITCH_69 DUR_2 ...  BAR ...  EOS

Unlike the notebook's 128-way multi-hot frames, every note is a discrete
(position, velocity, pitch, duration) group. That turns generation into ordinary
next-token classification, makes durations and repeated notes unambiguous, and
lets bars and beats be modelled directly.

The token grammar is simple enough to enforce while sampling (see
:class:`GrammarState`), so generated sequences always decode to valid music.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ..theory import Key, Note

# token types
PAD, BOS, EOS, BAR, POS, VEL, PITCH, DUR = range(8)
TYPE_NAMES = ("PAD", "BOS", "EOS", "BAR", "POS", "VEL", "PITCH", "DUR")


@dataclass(frozen=True)
class REMITokenizer:
    steps_per_bar: int = 16
    min_pitch: int = 21
    max_pitch: int = 108
    max_duration: int = 32
    velocity_bins: int = 8

    # ------------------------------------------------------------------ layout
    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def bar_id(self) -> int:
        return 3

    @property
    def pos_offset(self) -> int:
        return 4

    @property
    def vel_offset(self) -> int:
        return self.pos_offset + self.steps_per_bar

    @property
    def pitch_offset(self) -> int:
        return self.vel_offset + self.velocity_bins

    @property
    def n_pitches(self) -> int:
        return self.max_pitch - self.min_pitch + 1

    @property
    def dur_offset(self) -> int:
        return self.pitch_offset + self.n_pitches

    @property
    def vocab_size(self) -> int:
        return self.dur_offset + self.max_duration

    def token_types(self) -> np.ndarray:
        """Array mapping token id -> token type (PAD, BOS, ... DUR)."""
        types = np.empty(self.vocab_size, dtype=np.int64)
        types[: self.pos_offset] = [PAD, BOS, EOS, BAR]
        types[self.pos_offset : self.vel_offset] = POS
        types[self.vel_offset : self.pitch_offset] = VEL
        types[self.pitch_offset : self.dur_offset] = PITCH
        types[self.dur_offset :] = DUR
        return types

    # ---------------------------------------------------------- token helpers
    def pos_token(self, position: int) -> int:
        return self.pos_offset + position

    def vel_token(self, velocity: int) -> int:
        bin_ = min(self.velocity_bins - 1, (int(velocity) - 1) * self.velocity_bins // 127)
        return self.vel_offset + max(0, bin_)

    def pitch_token(self, pitch: int) -> int:
        if not self.min_pitch <= pitch <= self.max_pitch:
            raise ValueError(f"pitch {pitch} outside [{self.min_pitch}, {self.max_pitch}]")
        return self.pitch_offset + pitch - self.min_pitch

    def dur_token(self, duration: int) -> int:
        return self.dur_offset + int(np.clip(duration, 1, self.max_duration)) - 1

    def velocity_of(self, token: int) -> int:
        bin_ = token - self.vel_offset
        return int(round((bin_ + 0.5) * 127 / self.velocity_bins))

    def describe(self, token: int) -> str:
        t = self.token_types()[token]
        if t == POS:
            return f"POS_{token - self.pos_offset}"
        if t == VEL:
            return f"VEL_{token - self.vel_offset}"
        if t == PITCH:
            return f"PITCH_{token - self.pitch_offset + self.min_pitch}"
        if t == DUR:
            return f"DUR_{token - self.dur_offset + 1}"
        return TYPE_NAMES[t]

    # ---------------------------------------------------------------- encode
    def encode(
        self, notes: Iterable[Note], n_bars: int | None = None, add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        notes = sorted(n for n in notes if self.min_pitch <= n.pitch <= self.max_pitch)
        last_bar = max((n.start // self.steps_per_bar for n in notes), default=-1)
        n_bars = max(n_bars or 0, last_bar + 1)

        tokens = [self.bos_id] if add_bos else []
        idx = 0
        for bar in range(n_bars):
            tokens.append(self.bar_id)
            bar_end = (bar + 1) * self.steps_per_bar
            current_pos = None
            while idx < len(notes) and notes[idx].start < bar_end:
                note = notes[idx]
                pos = note.start - bar * self.steps_per_bar
                if pos != current_pos:
                    tokens.append(self.pos_token(pos))
                    current_pos = pos
                tokens += [
                    self.vel_token(note.velocity),
                    self.pitch_token(note.pitch),
                    self.dur_token(note.duration),
                ]
                idx += 1
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    # ---------------------------------------------------------------- decode
    def decode(self, tokens: Sequence[int]) -> list[Note]:
        """Decode tokens to notes. Ungrammatical fragments are skipped, never raised on."""
        types = self.token_types()
        notes: list[Note] = []
        bar, pos = -1, 0
        velocity, pitch = 80, None
        for tok in tokens:
            tok = int(tok)
            if not 0 <= tok < self.vocab_size:
                continue
            t = types[tok]
            if t == EOS:
                break
            if t == BAR:
                bar, pos, pitch = bar + 1, 0, None
            elif t == POS:
                pos, pitch = tok - self.pos_offset, None
            elif t == VEL:
                velocity = self.velocity_of(tok)
            elif t == PITCH:
                pitch = tok - self.pitch_offset + self.min_pitch
            elif t == DUR and pitch is not None:
                start = max(bar, 0) * self.steps_per_bar + pos
                notes.append(Note(start, pitch, tok - self.dur_offset + 1, velocity))
                pitch = None
        return notes

    def count_bars(self, tokens: Sequence[int]) -> int:
        return int(np.sum(np.asarray(tokens) == self.bar_id))

    # ----------------------------------------------------------- augmentation
    def transpose(self, tokens: np.ndarray, semitones: int) -> np.ndarray | None:
        """Shift every PITCH token; ``None`` if any note would leave the pitch range."""
        if semitones == 0:
            return tokens
        tokens = np.asarray(tokens)
        is_pitch = (tokens >= self.pitch_offset) & (tokens < self.dur_offset)
        shifted = tokens[is_pitch] + semitones
        if shifted.size and (
            shifted.min() < self.pitch_offset or shifted.max() >= self.dur_offset
        ):
            return None
        out = tokens.copy()
        out[is_pitch] = shifted
        return out

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "steps_per_bar": self.steps_per_bar,
            "min_pitch": self.min_pitch,
            "max_pitch": self.max_pitch,
            "max_duration": self.max_duration,
            "velocity_bins": self.velocity_bins,
        }

    @classmethod
    def from_dict(cls, data: dict) -> REMITokenizer:
        return cls(**data)


class GrammarState:
    """Tracks where a partially generated sequence is in the REMI grammar.

    ``allowed()`` returns a boolean mask over the vocabulary of tokens that keep
    the sequence well-formed. Optional constraints:

    * ``key`` - only pitches in the key's scale may be generated,
    * ``pitch_range`` - restrict the register,
    * ``monophonic`` - at most one note per position (for melody models).

    Grammar::

        BOS -> BAR
        BAR -> BAR | POS | EOS
        POS -> VEL -> PITCH -> DUR
        DUR -> VEL (another note at the same position, polyphonic only)
             | POS (strictly later in the bar) | BAR | EOS
    """

    def __init__(
        self,
        tokenizer: REMITokenizer,
        key: Key | None = None,
        pitch_range: tuple[int, int] | None = None,
        monophonic: bool = False,
    ) -> None:
        self.tok = tokenizer
        self.types = tokenizer.token_types()
        self.monophonic = monophonic
        self.last_type = None
        self.position = -1
        self.bars = 0

        pitch_ok = np.zeros(tokenizer.vocab_size, dtype=bool)
        for pitch in range(tokenizer.min_pitch, tokenizer.max_pitch + 1):
            if pitch_range and not pitch_range[0] <= pitch <= pitch_range[1]:
                continue
            if key is not None and not key.contains(pitch):
                continue
            pitch_ok[tokenizer.pitch_token(pitch)] = True
        if not pitch_ok.any():
            raise ValueError("Pitch constraints exclude every pitch")
        self._pitch_mask = pitch_ok
        self._type_masks = {t: self.types == t for t in range(len(TYPE_NAMES))}

    def update(self, token: int) -> None:
        t = int(self.types[int(token)])
        if t == BAR:
            self.bars += 1
            self.position = -1
        elif t == POS:
            self.position = int(token) - self.tok.pos_offset
        self.last_type = t

    def feed(self, tokens: Iterable[int]) -> GrammarState:
        for tok in tokens:
            self.update(tok)
        return self

    def allowed(self, allow_eos: bool = True) -> np.ndarray:
        m = self._type_masks
        last = self.last_type
        if last is None:
            mask = m[BOS].copy()
        elif last == BOS:
            mask = m[BAR].copy()
        elif last == POS:
            mask = m[VEL].copy()
        elif last == VEL:
            mask = self._pitch_mask.copy()
        elif last == PITCH:
            mask = m[DUR].copy()
        elif last in (BAR, DUR):
            mask = m[BAR] | self._later_positions()
            if last == DUR and not self.monophonic:
                mask = mask | m[VEL]
        else:  # EOS / PAD: nothing sensible follows
            mask = m[PAD].copy()
        if allow_eos and last in (BAR, DUR):
            mask = mask | m[EOS]
        return mask

    def _later_positions(self) -> np.ndarray:
        mask = np.zeros(self.tok.vocab_size, dtype=bool)
        first = self.tok.pos_offset + self.position + 1
        mask[first : self.tok.vel_offset] = True
        return mask
