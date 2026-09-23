"""MIDI <-> quantized note lists.

Two fixes over the original notebook live here:

* **Tempo-aware quantization.** The notebook called ``get_piano_roll(fs=steps_per_beat)``,
  but ``fs`` is *frames per second*, so "4 steps per beat" really meant "4 frames per
  second" regardless of tempo. We quantize against the MIDI file's own beat grid,
  so a sixteenth note is always one step no matter the tempo.
* **Actually monophonic melodies.** The notebook picked the highest-pitched
  instrument but kept all of its chords. We apply a skyline reduction (keep the
  highest note at every onset, truncate overlaps) so the melody is truly monophonic.

Files that are malformed in common ways (for example an out-of-range key-signature
meta event, like ``sample_28.mid`` in the original run) are repaired on load
instead of dropped.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import mido
import numpy as np
import pretty_midi

from ..theory import Note

logger = logging.getLogger(__name__)

MIDI_EXTENSIONS = (".mid", ".midi", ".MID", ".MIDI")


class MidiLoadError(RuntimeError):
    """Raised when a MIDI file cannot be parsed even after repair attempts."""


@dataclass
class Score:
    """A quantized piece of music."""

    notes: list[Note]
    steps_per_beat: int = 4
    beats_per_bar: int = 4
    tempo: float = 120.0
    name: str = ""
    meta: dict = field(default_factory=dict)
    bars: int | None = None  # explicit length (e.g. a slice); otherwise derived from onsets

    @property
    def steps_per_bar(self) -> int:
        return self.steps_per_beat * self.beats_per_bar

    @property
    def n_steps(self) -> int:
        return max((n.end for n in self.notes), default=0)

    @property
    def n_bars(self) -> int:
        """Bars spanned by note onsets (a final held note ringing over does not add a bar)."""
        if self.bars is not None:
            return self.bars
        return max((n.start // self.steps_per_bar + 1 for n in self.notes), default=0)

    def slice_bars(self, start_bar: int, end_bar: int) -> Score:
        """Notes whose onsets fall in ``[start_bar, end_bar)``, shifted to start at bar 0."""
        lo, hi = start_bar * self.steps_per_bar, end_bar * self.steps_per_bar
        notes = [
            Note(n.start - lo, n.pitch, min(n.duration, hi - n.start), n.velocity)
            for n in self.notes
            if lo <= n.start < hi
        ]
        return Score(notes, self.steps_per_beat, self.beats_per_bar, self.tempo, self.name,
                     bars=end_bar - start_bar)


# --------------------------------------------------------------------------- loading
_patch_lock = threading.Lock()


@contextlib.contextmanager
def _lenient_key_signatures() -> Iterator[None]:
    """Temporarily make mido decode invalid key signatures as C major instead of raising."""
    spec = mido.midifiles.meta.MetaSpec_key_signature
    original = spec.decode

    def decode(self, message, data):  # pragma: no cover - exercised on broken files only
        try:
            original(self, message, data)
        except mido.midifiles.meta.KeySignatureError:
            message.key = "C"

    with _patch_lock:
        spec.decode = decode
        try:
            yield
        finally:
            spec.decode = original


def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    """Load a MIDI file, repairing common corruptions (bad key signatures, out-of-range data)."""
    path = Path(path)
    with warnings.catch_warnings():
        # "Tempo, Key or Time signature change events found on non-zero tracks" is harmless.
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return pretty_midi.PrettyMIDI(str(path))
        except Exception as first_error:  # noqa: BLE001 - mido raises many exception types
            try:
                with _lenient_key_signatures():
                    midi = mido.MidiFile(str(path), clip=True)
                pm = pretty_midi.PrettyMIDI(mido_object=midi)
                logger.debug("Repaired %s (%s)", path.name, first_error)
                return pm
            except Exception as second_error:  # noqa: BLE001
                raise MidiLoadError(f"{path.name}: {first_error or second_error}") from second_error


def find_midi_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix in MIDI_EXTENSIONS and p.is_file())


# ----------------------------------------------------------------------- quantization
def _make_quantizer(pm: pretty_midi.PrettyMIDI, steps_per_beat: int):
    """Return a function mapping seconds -> fractional grid steps using the file's beat grid."""
    beats = pm.get_beats()
    if len(beats) < 2:
        tempo = _initial_tempo(pm)
        seconds_per_step = 60.0 / (tempo * steps_per_beat)
        return lambda t: np.asarray(t, dtype=np.float64) / seconds_per_step

    beat_idx = np.arange(len(beats), dtype=np.float64)
    first_period = beats[1] - beats[0]
    last_period = beats[-1] - beats[-2]

    def to_steps(t):
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        b = np.interp(t, beats, beat_idx)
        before, after = t < beats[0], t > beats[-1]
        b[before] = (t[before] - beats[0]) / first_period
        b[after] = beat_idx[-1] + (t[after] - beats[-1]) / last_period
        return b * steps_per_beat

    return to_steps


def _initial_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    _, tempi = pm.get_tempo_changes()
    tempo = round(float(tempi[0]), 2) if len(tempi) else 120.0
    return tempo if 20 <= tempo <= 400 else 120.0


def select_melody_instrument(
    instruments: Iterable[pretty_midi.Instrument], min_notes: int = 16
) -> pretty_midi.Instrument | None:
    """Pick the instrument most likely to carry the melody.

    Keeps the notebook's "highest average pitch" heuristic, but ignores drum tracks and
    near-empty tracks (the original crashed with ``np.mean([])`` on empty instruments and
    could select the drum kit).
    """
    candidates = [inst for inst in instruments if not inst.is_drum and inst.notes]
    if not candidates:
        return None
    busiest = max(len(inst.notes) for inst in candidates)
    threshold = min(min_notes, busiest)
    candidates = [inst for inst in candidates if len(inst.notes) >= threshold]
    return max(candidates, key=lambda inst: np.mean([n.pitch for n in inst.notes]))


def skyline(notes: Iterable[Note]) -> list[Note]:
    """Reduce polyphony to a monophonic line: highest pitch per onset, overlaps truncated."""
    top: dict[int, Note] = {}
    for note in notes:
        current = top.get(note.start)
        if current is None or note.pitch > current.pitch:
            top[note.start] = note
    ordered = [top[s] for s in sorted(top)]
    melody = []
    for note, nxt in zip(ordered, ordered[1:] + [None], strict=True):
        duration = note.duration if nxt is None else min(note.duration, nxt.start - note.start)
        melody.append(Note(note.start, note.pitch, max(1, duration), note.velocity))
    return melody


def quantize_midi(
    pm: pretty_midi.PrettyMIDI,
    steps_per_beat: int = 4,
    beats_per_bar: int = 4,
    melody_only: bool = True,
    min_pitch: int = 0,
    max_pitch: int = 127,
    max_duration: int | None = None,
    name: str = "",
) -> Score:
    """Convert a PrettyMIDI object into a quantized :class:`Score`."""
    if melody_only:
        inst = select_melody_instrument(pm.instruments)
        raw = inst.notes if inst is not None else []
    else:
        raw = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]

    notes: dict[tuple[int, int], Note] = {}
    if raw:
        to_steps = _make_quantizer(pm, steps_per_beat)
        starts = np.round(to_steps([n.start for n in raw])).astype(int)
        ends = np.round(to_steps([n.end for n in raw])).astype(int)
        for n, s, e in zip(raw, starts, ends, strict=True):
            if s < 0 or not min_pitch <= n.pitch <= max_pitch:
                continue
            duration = max(1, int(e - s))
            if max_duration:
                duration = min(duration, max_duration)
            key = (int(s), n.pitch)
            prev = notes.get(key)
            if prev is None or duration > prev.duration:  # de-duplicate doubled notes
                notes[key] = Note(int(s), n.pitch, duration, int(np.clip(n.velocity, 1, 127)))

    ordered = sorted(notes.values())
    if melody_only:
        ordered = skyline(ordered)
    return Score(ordered, steps_per_beat, beats_per_bar, _initial_tempo(pm), name)


def load_score(path: str | Path, **kwargs) -> Score:
    """Load and quantize a MIDI file in one call."""
    path = Path(path)
    return quantize_midi(load_midi(path), name=path.stem, **kwargs)


# --------------------------------------------------------------------------- writing
def notes_to_midi(
    notes: Iterable[Note],
    steps_per_beat: int = 4,
    tempo: float = 120.0,
    program: int = 0,
    name: str = "musegen",
) -> pretty_midi.PrettyMIDI:
    """Render quantized notes to a PrettyMIDI object.

    Each :class:`Note` becomes one sustained MIDI note. (The original notebook emitted a
    separate 0.25 s note for every active frame, so held notes were re-struck every step.)
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=program, name=name)
    seconds_per_step = 60.0 / (tempo * steps_per_beat)
    for note in sorted(notes):
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(np.clip(note.velocity, 1, 127)),
                pitch=int(note.pitch),
                start=note.start * seconds_per_step,
                end=note.end * seconds_per_step,
            )
        )
    pm.instruments.append(instrument)
    return pm


def save_score(score: Score, path: str | Path, program: int = 0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = notes_to_midi(score.notes, score.steps_per_beat, score.tempo, program, score.name or "musegen")
    pm.write(str(path))
    return path
