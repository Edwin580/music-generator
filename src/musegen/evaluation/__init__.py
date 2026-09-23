"""Checkpoint evaluation: held-out likelihood plus musical statistics of generated samples."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from ..config import GenerationConfig
from ..data.dataset import build_corpus, split_scores
from ..data.midi_io import save_score
from ..generation import generate
from ..models import load_checkpoint
from ..training import make_dataset, resolve_device
from ..training.trainer import evaluate_loader, load_history
from .metrics import aggregate, compare_sets, piece_metrics, pooled_histograms

__all__ = ["aggregate", "compare_sets", "evaluate_checkpoint", "piece_metrics"]

logger = logging.getLogger(__name__)

HEADLINE_METRICS = (
    "notes_per_bar", "pitch_range", "pitch_class_entropy", "scale_consistency", "key_clarity",
    "mean_abs_interval", "large_leap_rate", "repetition_ratio", "duration_entropy",
    "onset_position_entropy", "empty_bar_rate",
)


def evaluate_checkpoint(
    checkpoint: str | Path,
    out_dir: str | Path,
    n_samples: int = 16,
    bars: int = 16,
    prompt_bars: int = 4,
    gen_overrides: dict | None = None,
    device: str = "auto",
    seed: int = 0,
    save_midi: int = 4,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)
    model, cfg, tokenizer, _ = load_checkpoint(checkpoint, dev)
    gen: GenerationConfig = dataclasses.replace(cfg.generation, **(gen_overrides or {}),
                                                bars=bars)

    corpus = build_corpus(cfg.data, progress=False)
    splits = split_scores(corpus.scores, cfg.data.val_fraction, cfg.data.test_fraction,
                          cfg.data.seed)
    split_name = next(k for k in ("test", "val", "train") if splits[k])
    reference = splits[split_name]
    logger.info("Evaluating on the %s split (%d pieces)", split_name, len(reference))

    loader = DataLoader(make_dataset(cfg, reference, train=False),
                        batch_size=cfg.train.batch_size)
    likelihood = evaluate_loader(model, loader, dev, tokenizer.pad_id)

    generated, truth = [], []
    for i in range(n_samples):
        source = reference[i % len(reference)]
        prompt = source.slice_bars(0, prompt_bars) if prompt_bars else None
        result = generate(model, tokenizer, gen, prompt, monophonic=cfg.data.melody_only,
                          seed=seed + i, steps_per_beat=cfg.data.steps_per_beat)
        continuation = result.score.slice_bars(prompt_bars, prompt_bars + bars)
        generated.append(continuation)
        truth.append(source.slice_bars(prompt_bars, prompt_bars + bars))
        if i < save_midi:
            save_score(result.score, out_dir / f"sample_{i:02d}.mid", gen.program)
            if i == 0:
                from ..visualization import plot_piano_roll

                plot_piano_roll(result.score, out_dir / "sample_00_piano_roll.png",
                                title=f"Continuation of '{source.name}'",
                                prompt_bars=prompt_bars)
    truth = [s for s in truth if s.notes]

    report = {
        "checkpoint": str(checkpoint),
        "arch": cfg.model.arch,
        "split": split_name,
        "held_out_likelihood": likelihood,
        "generated": aggregate(generated),
        "reference": aggregate(truth),
        "distribution_similarity": compare_sets(generated, truth) if truth else {},
        "generation": dataclasses.asdict(gen),
        "n_samples": n_samples,
    }
    with open(out_dir / "report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    (out_dir / "report.md").write_text(render_markdown(report))

    from ..visualization import plot_distribution_comparison, plot_training_history

    if truth:
        plot_distribution_comparison(pooled_histograms(generated), pooled_histograms(truth),
                                     out_dir / "distributions.png")
    history_csv = Path(checkpoint).parent / "history.csv"
    if history_csv.exists():
        plot_training_history(load_history(history_csv), out_dir / "training_history.png")
    return report


def render_markdown(report: dict) -> str:
    lines = [f"# Evaluation: `{report['arch']}`", "",
             f"Checkpoint: `{report['checkpoint']}` | split: **{report['split']}** | "
             f"samples: {report['n_samples']}", "", "## Held-out likelihood", ""]
    lines += [f"- **{k}**: {v:.4f}" for k, v in report["held_out_likelihood"].items()]
    lines += ["", "## Musical statistics (generated vs. real continuation)", "",
              "| metric | generated | reference |", "|---|---|---|"]
    for key in HEADLINE_METRICS:
        g, r = report["generated"].get(key), report["reference"].get(key)
        if g and r:
            lines.append(f"| {key} | {g['mean']:.3f} ± {g['std']:.3f} | "
                         f"{r['mean']:.3f} ± {r['std']:.3f} |")
    if report["distribution_similarity"]:
        lines += ["", "## Distribution similarity", "",
                  "| feature | overlap area (↑) | JS divergence (↓) |", "|---|---|---|"]
        for name, vals in report["distribution_similarity"].items():
            lines.append(f"| {name} | {vals['overlap_area']:.3f} | {vals['js_divergence']:.3f} |")
    return "\n".join(lines) + "\n"
