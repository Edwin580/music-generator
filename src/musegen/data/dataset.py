"""Corpus loading, file-level splits and PyTorch datasets.

Fixes a subtle problem in the original notebook: it cut every song into windows with a
stride of one step, shuffled all windows together and then used
``validation_split=0.2``. Neighbouring windows overlap by 47 of 48 frames, so the
validation set was almost a copy of the training set and validation loss said little
about generalisation. Here splits are made **per file**, windows use a configurable
stride and start on bar lines, and training windows get random transposition as
augmentation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..config import DataConfig
from .midi_io import MidiLoadError, Score, find_midi_files, load_score
from .pianoroll import notes_to_roll, transpose_roll
from .tokenizer import REMITokenizer

logger = logging.getLogger(__name__)

CACHE_VERSION = 3
MIN_NOTES = 8


@dataclass
class Corpus:
    scores: list[Score]
    failures: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        n_notes = sum(len(s.notes) for s in self.scores)
        n_bars = sum(s.n_bars for s in self.scores)
        return (
            f"{len(self.scores)} pieces, {n_notes:,} notes, {n_bars:,} bars"
            f" ({len(self.failures)} files skipped)"
        )


def build_tokenizer(cfg: DataConfig) -> REMITokenizer:
    return REMITokenizer(
        steps_per_bar=cfg.steps_per_beat * cfg.beats_per_bar,
        min_pitch=cfg.min_pitch,
        max_pitch=cfg.max_pitch,
        max_duration=cfg.max_duration_steps,
        velocity_bins=cfg.velocity_bins,
    )


def _cache_key(files: list[Path], cfg: DataConfig) -> str:
    relevant = {
        k: v
        for k, v in asdict(cfg).items()
        if k in {"steps_per_beat", "beats_per_bar", "melody_only", "min_pitch", "max_pitch",
                 "max_duration_steps"}
    }
    listing = [(str(f), f.stat().st_size, int(f.stat().st_mtime)) for f in files]
    blob = json.dumps({"v": CACHE_VERSION, "cfg": relevant, "files": listing}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def build_corpus(cfg: DataConfig, midi_dir: str | Path | None = None, use_cache: bool = True,
                 progress: bool = True) -> Corpus:
    """Load and quantize every MIDI file under ``midi_dir`` (cached on disk)."""
    midi_dir = Path(midi_dir or cfg.midi_dir)
    files = find_midi_files(midi_dir)
    if not files:
        raise FileNotFoundError(
            f"No MIDI files found in {midi_dir}. Run `musegen download` or "
            f"`musegen synth-data`, or point data.midi_dir at your own collection."
        )

    cache_path = Path(cfg.cache_dir) / f"corpus_{_cache_key(files, cfg)}.pkl"
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as fh:
            corpus = pickle.load(fh)
        logger.info("Loaded cached corpus from %s: %s", cache_path, corpus.summary())
        return corpus

    scores, failures = [], []
    for path in tqdm(files, desc="Parsing MIDI", disable=not progress):
        try:
            score = load_score(
                path,
                steps_per_beat=cfg.steps_per_beat,
                beats_per_bar=cfg.beats_per_bar,
                melody_only=cfg.melody_only,
                min_pitch=cfg.min_pitch,
                max_pitch=cfg.max_pitch,
                max_duration=cfg.max_duration_steps,
            )
        except MidiLoadError as exc:
            failures.append((path.name, str(exc)))
            logger.warning("Skipping %s", exc)
            continue
        if len(score.notes) < MIN_NOTES:
            failures.append((path.name, f"only {len(score.notes)} notes"))
            continue
        scores.append(score)

    if not scores:
        raise ValueError(f"None of the {len(files)} MIDI files in {midi_dir} were usable")
    corpus = Corpus(scores, failures)
    logger.info("Built corpus: %s", corpus.summary())
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(corpus, fh)
    return corpus


def split_scores(
    scores: list[Score], val_fraction: float, test_fraction: float, seed: int = 42
) -> dict[str, list[Score]]:
    """Deterministic per-file train/val/test split."""
    ordered = sorted(scores, key=lambda s: s.name)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    n = len(ordered)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    if n >= 3:
        n_val = max(n_val, 1 if val_fraction > 0 else 0)
    n_test = min(n_test, max(0, n - n_val - 1))
    test = [ordered[i] for i in perm[:n_test]]
    val = [ordered[i] for i in perm[n_test : n_test + n_val]]
    train = [ordered[i] for i in perm[n_test + n_val :]]
    return {"train": train, "val": val, "test": test}


# ----------------------------------------------------------------- datasets
def _random_shift(transpose_range: int) -> int:
    if transpose_range <= 0:
        return 0
    return int(torch.randint(-transpose_range, transpose_range + 1, ()).item())


class TokenWindowDataset(Dataset):
    """Fixed-length, bar-aligned windows over REMI token sequences for next-token prediction."""

    def __init__(
        self,
        scores: list[Score],
        tokenizer: REMITokenizer,
        seq_len: int,
        stride: int,
        transpose_range: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.transpose_range = transpose_range
        self.sequences = [np.asarray(tokenizer.encode(s.notes), dtype=np.int64) for s in scores]
        self.windows: list[tuple[int, int]] = []
        for i, seq in enumerate(self.sequences):
            bar_starts = np.flatnonzero(seq == tokenizer.bar_id)
            for start in _window_starts(len(seq), seq_len + 1, stride, bar_starts):
                self.windows.append((i, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i, start = self.windows[idx]
        chunk = self.sequences[i][start : start + self.seq_len + 1]
        shifted = self.tokenizer.transpose(chunk, _random_shift(self.transpose_range))
        if shifted is not None:
            chunk = shifted
        if len(chunk) < self.seq_len + 1:
            pad = np.full(self.seq_len + 1 - len(chunk), self.tokenizer.pad_id, dtype=np.int64)
            chunk = np.concatenate([chunk, pad])
        chunk = torch.from_numpy(np.ascontiguousarray(chunk))
        return chunk[:-1], chunk[1:]

    @property
    def n_tokens(self) -> int:
        return int(sum(len(s) for s in self.sequences))


class PianoRollWindowDataset(Dataset):
    """Windows of (sustain, onset) piano-roll frames; target is the next frame."""

    def __init__(
        self,
        scores: list[Score],
        min_pitch: int,
        max_pitch: int,
        seq_len: int,
        stride: int,
        transpose_range: int = 0,
    ) -> None:
        self.seq_len = seq_len
        self.transpose_range = transpose_range
        self.rolls = [notes_to_roll(s.notes, min_pitch, max_pitch) for s in scores]
        self.windows = [
            (i, start)
            for i, roll in enumerate(self.rolls)
            for start in _window_starts(len(roll), seq_len + 1, stride)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i, start = self.windows[idx]
        chunk = self.rolls[i][start : start + self.seq_len + 1]
        shifted = transpose_roll(chunk, _random_shift(self.transpose_range))
        if shifted is not None:
            chunk = shifted
        if len(chunk) < self.seq_len + 1:
            pad = np.zeros((self.seq_len + 1 - len(chunk), chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
        chunk = torch.from_numpy(np.ascontiguousarray(chunk))
        return chunk[:-1], chunk[1:]


def _window_starts(
    length: int, window: int, stride: int, anchors: np.ndarray | None = None
) -> list[int]:
    """Window start offsets covering a sequence. Starts snap forward to the next anchor
    (bar line) where possible; the last window is always flush with the end."""
    if length < 2:
        return []
    if length <= window:
        return [0]
    starts, s = [], 0
    last = length - window
    while s < last:
        starts.append(s)
        target = s + max(1, stride)
        if anchors is not None and len(anchors):
            nxt = anchors[anchors >= target]
            target = int(nxt[0]) if len(nxt) else last
        s = min(target, last)
    starts.append(last)
    return starts
