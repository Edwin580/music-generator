"""Piano-roll representation (the notebook's original encoding, made lossless).

The notebook used a single binary "is this pitch sounding" matrix. That cannot tell
one long note from several repeated short ones, which is why every generated note
was re-struck each step. Here each frame has two halves:

* ``frame[:, :P]`` - pitch is sounding (sustain)
* ``frame[:, P:]`` - pitch starts on this step (onset)

so notes round-trip exactly through :func:`notes_to_roll` / :func:`roll_to_notes`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..theory import Note


def notes_to_roll(
    notes: Iterable[Note], min_pitch: int, max_pitch: int, n_steps: int | None = None
) -> np.ndarray:
    notes = [n for n in notes if min_pitch <= n.pitch <= max_pitch]
    n_pitches = max_pitch - min_pitch + 1
    if n_steps is None:
        n_steps = max((n.end for n in notes), default=0)
    roll = np.zeros((n_steps, 2 * n_pitches), dtype=np.float32)
    for note in notes:
        if note.start >= n_steps:
            continue
        p = note.pitch - min_pitch
        roll[note.start : min(note.end, n_steps), p] = 1.0
        roll[note.start, n_pitches + p] = 1.0
    return roll


def roll_to_notes(roll: np.ndarray, min_pitch: int, velocity: int = 80) -> list[Note]:
    """Decode a (T, 2P) roll. A note starts at an onset (or where sustain begins without
    one) and lasts until sustain stops or the pitch is re-struck."""
    roll = np.asarray(roll) > 0.5
    n_steps, width = roll.shape
    n_pitches = width // 2
    sustain, onset = roll[:, :n_pitches], roll[:, n_pitches:]
    notes: list[Note] = []
    for p in range(n_pitches):
        start = None
        for t in range(n_steps + 1):
            active = t < n_steps and (sustain[t, p] or onset[t, p])
            restrike = t < n_steps and onset[t, p]
            if start is not None and (not active or restrike):
                notes.append(Note(start, p + min_pitch, t - start, velocity))
                start = None
            if active and start is None:
                start = t
    return sorted(notes)


def transpose_roll(roll: np.ndarray, semitones: int) -> np.ndarray | None:
    """Shift a (T, 2P) roll by ``semitones``; ``None`` if active notes would fall off."""
    if semitones == 0:
        return roll
    n_pitches = roll.shape[1] // 2
    out = np.zeros_like(roll)
    for half in (0, n_pitches):
        block = roll[:, half : half + n_pitches]
        if semitones > 0:
            if block[:, n_pitches - semitones :].any():
                return None
            out[:, half + semitones : half + n_pitches] = block[:, : n_pitches - semitones]
        else:
            s = -semitones
            if block[:, :s].any():
                return None
            out[:, half : half + n_pitches - s] = block[:, s:]
    return out
