"""Audio rendering.

The notebook used ``PrettyMIDI.synthesize()`` (pure sine waves, hence the "beeps") and
wrote the WAV samples to a file called ``output_test.mp3``. Here:

1. If ``pyfluidsynth`` + FluidSynth are installed, render with a real SoundFont
   (pretty_midi ships a small General MIDI one), so the piano sounds like a piano.
2. Otherwise use a built-in additive synthesizer with piano-like partials, per-partial
   exponential decay, hammer attack and release envelopes, which sounds far better than
   bare sine waves and needs no extra dependencies.

Output is 16-bit PCM WAV written with the standard library.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np
import pretty_midi

logger = logging.getLogger(__name__)

# (relative amplitude, decay rate multiplier) for harmonics 1..8 - brighter partials die faster
_PARTIALS = [(1.0, 1.0), (0.55, 1.4), (0.32, 1.9), (0.22, 2.3), (0.14, 2.8), (0.09, 3.3),
             (0.06, 3.9), (0.04, 4.5)]


def _additive_note(freq: float, duration: float, velocity: int, sr: int) -> np.ndarray:
    release = 0.12
    n = int((duration + release) * sr)
    t = np.arange(n) / sr
    # low notes ring longer than high notes, as on a real piano
    base_decay = 0.6 + 2.2 * (freq / 1000.0)
    wave_ = np.zeros(n)
    for k, (amp, decay) in enumerate(_PARTIALS, start=1):
        f = freq * k * np.sqrt(1 + 0.0004 * k * k)  # slight inharmonicity of piano strings
        if f >= sr / 2:
            break
        wave_ += amp * np.exp(-base_decay * decay * t) * np.sin(2 * np.pi * f * t)
    attack = np.minimum(1.0, t / 0.004)
    held = int(duration * sr)
    env = attack.copy()
    env[held:] *= np.exp(-(t[held:] - duration) / (release / 4))
    return (velocity / 127.0) ** 1.5 * wave_ * env


def synthesize(pm: pretty_midi.PrettyMIDI, sample_rate: int = 44100) -> np.ndarray:
    length = int((pm.get_end_time() + 0.5) * sample_rate)
    out = np.zeros(max(length, 1))
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            wave_ = _additive_note(pretty_midi.note_number_to_hz(note.pitch),
                                   max(note.end - note.start, 0.02), note.velocity, sample_rate)
            start = int(note.start * sample_rate)
            end = min(start + len(wave_), len(out))
            out[start:end] += wave_[: end - start]
    return out


def render(pm: pretty_midi.PrettyMIDI, sample_rate: int = 44100,
           soundfont: str | None = None, prefer_fluidsynth: bool = True) -> np.ndarray:
    """Render MIDI to a float waveform in [-1, 1]."""
    audio = None
    if prefer_fluidsynth:
        try:
            audio = pm.fluidsynth(fs=sample_rate, sf2_path=soundfont) if soundfont else \
                pm.fluidsynth(fs=sample_rate)
        except Exception as exc:  # noqa: BLE001 - ImportError or FluidSynth runtime errors
            logger.debug("FluidSynth unavailable (%s); using built-in synthesizer", exc)
    if audio is None or not np.any(audio):
        audio = synthesize(pm, sample_rate)
    peak = np.max(np.abs(audio)) if audio.size else 0
    return audio / peak * 0.9 if peak > 0 else audio


def write_wav(audio: np.ndarray, path: str | Path, sample_rate: int = 44100) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


def midi_to_wav(midi_path: str | Path, wav_path: str | Path, sample_rate: int = 44100,
                soundfont: str | None = None) -> Path:
    from .data.midi_io import load_midi

    return write_wav(render(load_midi(midi_path), sample_rate, soundfont), wav_path, sample_rate)
