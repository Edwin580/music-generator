"""Objective musical metrics for generated music.

The notebook compared one generated clip against one 48-step training window by eye.
Here we compute per-piece descriptors and compare whole *sets* of generated pieces
against held-out real music, in the spirit of Yang & Lerch (2020), "On the evaluation
of generative models in music":

* per-piece: pitch range / entropy, scale consistency, key clarity, note density,
  interval size, rhythmic variety, repetition (motif re-use), polyphony, empty bars;
* set-vs-set: overlap area and Jensen-Shannon divergence between the pooled
  histograms of pitch class, melodic interval, duration and onset position.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from ..data.midi_io import Score
from ..theory import Key, estimate_key, key_correlations, pitch_class_histogram


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _melodic_line(score: Score) -> list:
    """Highest note per onset: the line a listener tracks as melody."""
    top = {}
    for n in score.notes:
        if n.start not in top or n.pitch > top[n.start].pitch:
            top[n.start] = n
    return [top[s] for s in sorted(top)]


def scale_consistency(score: Score) -> float:
    """Fraction of notes inside the best-fitting major / (harmonic-)minor scale."""
    if not score.notes:
        return 0.0
    best = 0.0
    for mode in ("major", "minor"):
        for tonic in range(12):
            key = Key(tonic, mode)
            best = max(best, np.mean([key.contains(n.pitch) for n in score.notes]))
    return float(best)


def repetition_ratio(score: Score, n: int = 4) -> float:
    """Share of n-note (interval, duration) patterns that occur more than once.

    Transposition-invariant, so a motif restated a third higher still counts, which is
    a crude but useful proxy for the phrase-level structure that makes music sound composed.
    """
    line = _melodic_line(score)
    if len(line) <= n:
        return 0.0
    events = [(b.pitch - a.pitch, a.duration) for a, b in zip(line, line[1:], strict=False)]
    grams = [tuple(events[i : i + n]) for i in range(len(events) - n + 1)]
    counts = Counter(grams)
    return float(sum(c for c in counts.values() if c > 1) / len(grams))


def piece_metrics(score: Score) -> dict[str, float]:
    notes = score.notes
    if not notes:
        return {"n_notes": 0.0}
    pitches = np.array([n.pitch for n in notes])
    line = _melodic_line(score)
    intervals = np.abs(np.diff([n.pitch for n in line])) if len(line) > 1 else np.array([0])
    spb = score.steps_per_bar
    n_bars = max(1, score.n_bars)
    bars_with_notes = len({n.start // spb for n in notes})
    onsets = Counter(n.start for n in notes)
    corr = key_correlations(pitch_class_histogram(notes))
    key = estimate_key(notes)
    return {
        "n_notes": float(len(notes)),
        "n_bars": float(n_bars),
        "pitch_mean": float(pitches.mean()),
        "pitch_range": float(pitches.max() - pitches.min()),
        "pitch_classes_used": float(len(set(pitches % 12))),
        "pitch_class_entropy": _entropy(pitch_class_histogram(notes, weighted=False)),
        "scale_consistency": scale_consistency(score),
        "key_clarity": float(corr.max()),  # correlation with best Krumhansl-Kessler profile
        "key_is_minor": float(key is not None and key.mode == "minor"),
        "notes_per_bar": len(notes) / n_bars,
        "mean_duration": float(np.mean([n.duration for n in notes])),
        "duration_entropy": _entropy(np.bincount([n.duration for n in notes])),
        "onset_position_entropy": _entropy(np.bincount([n.start % spb for n in notes],
                                                       minlength=spb)),
        "mean_abs_interval": float(intervals.mean()),
        "large_leap_rate": float(np.mean(intervals > 7)),
        "repetition_ratio": repetition_ratio(score),
        "polyphony_rate": float(np.mean([c > 1 for c in onsets.values()])),
        "empty_bar_rate": 1.0 - bars_with_notes / n_bars,
    }


def aggregate(scores: Sequence[Score]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of every per-piece metric across a set."""
    rows = [piece_metrics(s) for s in scores if s.notes]
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows]))} for k in keys}


# ---------------------------------------------------------------- set comparisons
def _histograms(scores: Sequence[Score]) -> dict[str, np.ndarray]:
    spb = scores[0].steps_per_bar if scores else 16
    pc = np.zeros(12)
    interval = np.zeros(25)  # -12..+12 semitones (clipped)
    duration = np.zeros(33)
    onset = np.zeros(spb)
    for s in scores:
        pc += pitch_class_histogram(s.notes, weighted=False)
        line = _melodic_line(s)
        for a, b in zip(line, line[1:], strict=False):
            interval[int(np.clip(b.pitch - a.pitch, -12, 12)) + 12] += 1
        for n in s.notes:
            duration[min(n.duration, 32)] += 1
            onset[n.start % spb] += 1
    return {"pitch_class": pc, "interval": interval, "duration": duration, "onset_position": onset}


def _normalise(h: np.ndarray) -> np.ndarray:
    total = h.sum()
    return h / total if total > 0 else np.full_like(h, 1.0 / len(h))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = _normalise(p), _normalise(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def overlap_area(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.minimum(_normalise(p), _normalise(q)).sum())


def compare_sets(generated: Sequence[Score], reference: Sequence[Score]) -> dict[str, dict]:
    """Distribution similarity between generated and reference music (1.0 OA = identical)."""
    hg, hr = _histograms(generated), _histograms(reference)
    return {
        name: {"overlap_area": overlap_area(hg[name], hr[name]),
               "js_divergence": js_divergence(hg[name], hr[name])}
        for name in hg
    }


def pooled_histograms(scores: Sequence[Score]) -> dict[str, np.ndarray]:
    return {k: _normalise(v) for k, v in _histograms(scores).items()}
