"""Figures: piano rolls, training curves and distribution comparisons.

The notebook drew piano rolls as scatter points (one dot per active frame) with
plotnine. Here notes are drawn as bars with true durations, the prompt and the
continuation are shaded differently, and bar lines are marked.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data.midi_io import Score  # noqa: E402
from .theory import PITCH_CLASS_NAMES, pitch_name  # noqa: E402

BLUE, RED, GREEN, PURPLE, GREY = "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#9ca3af"


def _finish(fig, path: str | Path | None):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_piano_roll(score: Score, path: str | Path | None = None, title: str = "Piano roll",
                    prompt_bars: int = 0, max_bars: int | None = None, ax=None):
    own = ax is None
    fig = plt.figure(figsize=(14, 5)) if own else ax.figure
    ax = ax or fig.add_subplot(111)
    spb = score.steps_per_bar
    limit = max_bars * spb if max_bars else None
    notes = [n for n in score.notes if limit is None or n.start < limit]
    if not notes:
        ax.text(0.5, 0.5, "no notes", ha="center", va="center", transform=ax.transAxes)
        return _finish(fig, path) if own else ax

    prompt_end = prompt_bars * spb
    for n in notes:
        color = BLUE if n.start < prompt_end else RED
        alpha = 0.35 + 0.65 * n.velocity / 127
        ax.broken_barh([(n.start, n.duration)], (n.pitch - 0.4, 0.8), facecolors=color,
                       alpha=alpha, edgecolor="white", linewidth=0.5)
    end = max(n.end for n in notes)
    for bar in range(0, end + spb, spb):
        ax.axvline(bar, color=GREY, lw=0.6 if bar % (4 * spb) else 1.2, alpha=0.6, zorder=0)
    if prompt_end:
        ax.axvspan(0, prompt_end, color=BLUE, alpha=0.06, zorder=0)
        ax.text(prompt_end, 1.01, " generated →", transform=ax.get_xaxis_transform(),
                color=RED, fontsize=9, va="bottom")
    lo, hi = min(n.pitch for n in notes), max(n.pitch for n in notes)
    ticks = [p for p in range(lo - 2, hi + 3) if p % 12 in (0, 7)]
    ax.set_yticks(ticks, [pitch_name(p) for p in ticks])
    ax.set_ylim(lo - 2, hi + 2)
    ax.set_xlim(0, end)
    xt = list(range(0, end + 1, spb * max(1, (end // spb) // 16)))
    ax.set_xticks(xt, [str(t // spb + 1) for t in xt])
    ax.set_xlabel("Bar")
    ax.set_ylabel("Pitch")
    ax.set_title(title)
    return _finish(fig, path) if own else ax


def plot_training_history(history: Sequence[dict], path: str | Path | None = None):
    epochs = [row["epoch"] for row in history]
    token_model = any("train_perplexity" in row for row in history)
    metric = "perplexity" if token_model else "f1"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in ((axes[0], "loss", "Loss"),
                           (axes[1], metric, metric.capitalize() if token_model else "Frame F1")):
        for split, color in (("train", BLUE), ("val", RED)):
            values = [row.get(f"{split}_{key}") for row in history]
            if any(v is not None for v in values):
                ax.plot(epochs, values, "o-", color=color, label=split.capitalize(), lw=2)
        ax.set_xlabel("Epoch")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[2].plot(epochs, [row.get("lr") for row in history], "o-", color=GREEN, lw=2)
    axes[2].set_title("Learning rate (end of epoch)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)
    return _finish(fig, path)


def plot_distribution_comparison(generated: dict[str, np.ndarray], reference: dict[str, np.ndarray],
                                 path: str | Path | None = None):
    panels = [
        ("pitch_class", "Pitch class", list(PITCH_CLASS_NAMES)),
        ("interval", "Melodic interval (semitones)", [str(i) for i in range(-12, 13)]),
        ("duration", "Duration (steps)", [str(i) for i in range(33)]),
        ("onset_position", "Onset position in bar", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (key, title, labels) in zip(axes.flat, panels, strict=True):
        ref, gen = reference[key], generated[key]
        x = np.arange(len(ref))
        ax.bar(x - 0.2, ref, 0.4, color=BLUE, alpha=0.8, label="Reference (held-out)")
        ax.bar(x + 0.2, gen, 0.4, color=RED, alpha=0.8, label="Generated")
        if labels:
            step = 1 if len(labels) <= 13 else 4
            ax.set_xticks(x[::step], labels[::step])
        ax.set_title(title)
        ax.set_ylabel("Share")
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend()
    return _finish(fig, path)


def plot_key_profile(score: Score, path: str | Path | None = None):
    """Duration-weighted pitch-class profile against the best-matching key profile."""
    from .theory import MAJOR_PROFILE, MINOR_PROFILE, estimate_key, pitch_class_histogram

    hist = pitch_class_histogram(score.notes)
    key = estimate_key(score.notes)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(12)
    ax.bar(x, hist / max(hist.sum(), 1e-9), color=BLUE, alpha=0.8, label="Piece")
    if key is not None:
        profile = np.roll(MAJOR_PROFILE if key.mode == "major" else MINOR_PROFILE, key.tonic)
        ax.plot(x, profile / profile.sum(), "o-", color=RED, lw=2,
                label=f"Krumhansl-Kessler: {key.name} (r={key.confidence:.2f})")
    ax.set_xticks(x, PITCH_CLASS_NAMES)
    ax.set_ylabel("Share of duration")
    ax.set_title("Tonal profile")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    return _finish(fig, path)
