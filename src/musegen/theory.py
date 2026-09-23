"""Minimal music-theory toolkit: notes, keys, scales and key estimation.

Key estimation uses the Krumhansl-Schmuckler algorithm: the duration-weighted
pitch-class histogram of a passage is correlated with the Krumhansl-Kessler
probe-tone profiles (Krumhansl & Kessler, 1982) rotated to all 24 major/minor
keys, and the best-correlated key wins. These profiles come from listener
ratings of how well each pitch class "fits" a key, so this is a cognitive model
of tonality rather than a rule-based one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLAT_ALIASES = {"DB": 1, "EB": 3, "GB": 6, "AB": 8, "BB": 10, "CB": 11, "FB": 4}

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
# Harmonic-minor leading tone is common in minor-key melodies, so it is allowed too.
MINOR_SCALE_EXTENDED = (0, 2, 3, 5, 7, 8, 10, 11)


@dataclass(frozen=True, order=True)
class Note:
    """A note on a quantized grid. ``start`` and ``duration`` are measured in grid steps."""

    start: int
    pitch: int
    duration: int
    velocity: int = 80

    @property
    def end(self) -> int:
        return self.start + self.duration


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class 0-11
    mode: str  # "major" | "minor"
    confidence: float = 1.0  # Pearson correlation with the key profile

    def __post_init__(self) -> None:
        if self.mode not in ("major", "minor"):
            raise ValueError(f"mode must be 'major' or 'minor', got {self.mode!r}")
        if not 0 <= self.tonic < 12:
            raise ValueError("tonic must be a pitch class in [0, 11]")

    @property
    def name(self) -> str:
        return f"{PITCH_CLASS_NAMES[self.tonic]} {self.mode}"

    def pitch_classes(self) -> frozenset[int]:
        scale = MAJOR_SCALE if self.mode == "major" else MINOR_SCALE_EXTENDED
        return frozenset((self.tonic + step) % 12 for step in scale)

    def contains(self, pitch: int) -> bool:
        return pitch % 12 in self.pitch_classes()

    def __str__(self) -> str:
        return self.name

    @classmethod
    def parse(cls, text: str) -> Key:
        """Parse strings such as ``"C major"``, ``"f# minor"``, ``"Bb"``, ``"Am"``."""
        match = re.fullmatch(r"\s*([A-Ga-g])([#b]?)\s*(major|minor|maj|min|m)?\s*", text)
        if not match:
            raise ValueError(f"Cannot parse key {text!r}; try e.g. 'C major' or 'A minor'")
        letter, accidental, mode = match.groups()
        name = (letter + accidental).upper()
        tonic = _FLAT_ALIASES.get(name)
        if tonic is None:
            tonic = PITCH_CLASS_NAMES.index(name)
        mode = "minor" if mode in ("minor", "min", "m") else "major"
        return cls(tonic=tonic, mode=mode)


def pitch_class_histogram(notes: Iterable[Note], weighted: bool = True) -> np.ndarray:
    """12-bin pitch-class histogram, optionally weighted by note duration."""
    hist = np.zeros(12, dtype=np.float64)
    for note in notes:
        hist[note.pitch % 12] += note.duration if weighted else 1.0
    return hist


def key_correlations(histogram: Sequence[float]) -> np.ndarray:
    """Pearson correlation of a pitch-class histogram with all 24 keys.

    Returns an array of shape (2, 12): row 0 = major keys, row 1 = minor keys,
    column = tonic pitch class.
    """
    hist = np.asarray(histogram, dtype=np.float64)
    out = np.zeros((2, 12))
    if hist.sum() == 0 or np.allclose(hist, hist[0]):
        return out
    for mode_idx, profile in enumerate((MAJOR_PROFILE, MINOR_PROFILE)):
        for tonic in range(12):
            out[mode_idx, tonic] = np.corrcoef(hist, np.roll(profile, tonic))[0, 1]
    return out


def estimate_key(notes: Iterable[Note]) -> Key | None:
    """Krumhansl-Schmuckler key estimate, or ``None`` for empty/atonal input."""
    corr = key_correlations(pitch_class_histogram(notes))
    if not np.any(corr):
        return None
    mode_idx, tonic = np.unravel_index(np.argmax(corr), corr.shape)
    return Key(tonic=int(tonic), mode="major" if mode_idx == 0 else "minor",
               confidence=float(corr[mode_idx, tonic]))


def pitch_name(pitch: int) -> str:
    """MIDI pitch number -> scientific pitch name (60 -> 'C4')."""
    return f"{PITCH_CLASS_NAMES[pitch % 12]}{pitch // 12 - 1}"
