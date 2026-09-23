"""Dataset acquisition: the notebook's MIDIWorld sample set and a synthetic corpus.

The synthetic corpus lets the whole pipeline (and the test-suite / CI) run offline.
Its pieces are not random noise: each has a key, a chord progression, an AABA-style
phrase structure with a recurring motif, a bass/chord accompaniment and a drum
track, so melody extraction, key estimation and the models all have real structure
to find.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import pretty_midi

from ..theory import MAJOR_SCALE, NATURAL_MINOR_SCALE
from .midi_io import MidiLoadError, load_midi

logger = logging.getLogger(__name__)

# The sample set used by the original Colab notebook.
MIDIWORLD_IDS = (
    list(range(3832, 3841))
    + list(range(4537, 4566))
    + list(range(4573, 4593))
)
MIDIWORLD_URL = "https://www.midiworld.com/download/{id}"


def download_midiworld(dest: str | Path, retries: int = 3, timeout: float = 30.0) -> list[Path]:
    """Download the notebook's MIDIWorld sample files, validating each one parses."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for file_id in MIDIWORLD_IDS:
        path = dest / f"midiworld_{file_id}.mid"
        if path.exists():
            saved.append(path)
            continue
        url = MIDIWORLD_URL.format(id=file_id)
        for attempt in range(1, retries + 1):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "musegen/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as resp:
                    payload = resp.read()
                if not payload.startswith(b"MThd"):
                    raise ValueError("response is not a MIDI file")
                path.write_bytes(payload)
                load_midi(path)
                saved.append(path)
                logger.info("Downloaded %s", path.name)
                break
            except (OSError, ValueError, MidiLoadError) as exc:
                path.unlink(missing_ok=True)
                if attempt == retries:
                    logger.warning("Giving up on %s: %s", url, exc)
                else:
                    time.sleep(2**attempt)
    logger.info("%d/%d MIDIWorld files available in %s", len(saved), len(MIDIWORLD_IDS), dest)
    return saved


# ---------------------------------------------------------------- synthetic corpus
_RHYTHMS = [  # one bar of 16th-note steps: (onset, duration) pairs
    [(0, 4), (4, 4), (8, 4), (12, 4)],
    [(0, 2), (2, 2), (4, 4), (8, 2), (10, 2), (12, 4)],
    [(0, 6), (6, 2), (8, 8)],
    [(0, 3), (3, 1), (4, 4), (8, 3), (11, 1), (12, 4)],
    [(0, 4), (4, 2), (6, 2), (8, 4), (12, 2), (14, 2)],
    [(0, 8), (8, 4), (12, 4)],
]
_PROGRESSIONS = [(0, 5, 3, 4), (0, 3, 4, 0), (0, 4, 5, 3), (5, 3, 0, 4), (0, 5, 1, 4)]


def _synth_piece(rng: np.random.Generator, n_bars: int = 16) -> pretty_midi.PrettyMIDI:
    minor = rng.random() < 0.35
    scale = NATURAL_MINOR_SCALE if minor else MAJOR_SCALE
    tonic = int(rng.integers(0, 12))
    tempo = float(rng.choice([90, 100, 110, 120, 132, 140]))
    spb = 60.0 / tempo / 4  # seconds per 16th
    progression = _PROGRESSIONS[int(rng.integers(len(_PROGRESSIONS)))]
    base = 60 + tonic if tonic < 7 else 48 + tonic

    def degree_to_pitch(deg: int, octave_base: int) -> int:
        return octave_base + 12 * (deg // 7) + scale[deg % 7]

    # A motif is a (rhythm, contour) pair reused across the phrase structure.
    def make_phrase() -> list[tuple[int, int, int]]:
        phrase, deg = [], int(rng.integers(0, 5))
        for bar in range(4):
            chord_root = progression[bar % 4]
            for onset, dur in _RHYTHMS[int(rng.integers(len(_RHYTHMS)))]:
                if onset == 0:  # land on a chord tone at the downbeat
                    deg = chord_root + int(rng.choice([0, 2, 4]))
                else:
                    deg += int(rng.choice([-2, -1, -1, 1, 1, 2, 0, 3]))
                deg = int(np.clip(deg, -3, 11))
                phrase.append((bar * 16 + onset, dur, deg))
        return phrase

    a, b = make_phrase(), make_phrase()
    form = [a, a, b, a] if n_bars >= 16 else [a, b]

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    lead = pretty_midi.Instrument(program=int(rng.choice([0, 40, 73, 80])), name="lead")
    chords = pretty_midi.Instrument(program=0, name="chords")
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")

    for phrase_idx, phrase in enumerate(form):
        offset = phrase_idx * 64
        for onset, dur, deg in phrase:
            pitch = degree_to_pitch(deg, base)
            vel = int(rng.integers(70, 110)) if onset % 16 == 0 else int(rng.integers(55, 95))
            lead.notes.append(pretty_midi.Note(vel, pitch, (offset + onset) * spb,
                                               (offset + onset + dur) * spb))
        for bar in range(4):
            root = progression[bar % 4]
            start, end = (offset + bar * 16) * spb, (offset + bar * 16 + 16) * spb
            for chord_deg in (root, root + 2, root + 4):
                chords.notes.append(pretty_midi.Note(60, degree_to_pitch(chord_deg, base - 24),
                                                     start, end))
            for beat in range(4):
                t = (offset + bar * 16 + beat * 4) * spb
                drum_pitch = 36 if beat % 2 == 0 else 38
                drums.notes.append(pretty_midi.Note(90, drum_pitch, t, t + spb))
    pm.instruments.extend([lead, chords, drums])
    return pm


def generate_synthetic_corpus(dest: str | Path, n_pieces: int = 64, seed: int = 0) -> list[Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_pieces):
        path = dest / f"synthetic_{i:04d}.mid"
        _synth_piece(rng).write(str(path))
        paths.append(path)
    return paths
