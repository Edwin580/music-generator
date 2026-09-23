"""End-to-end training pipeline."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import ExperimentConfig
from ..data.dataset import (
    PianoRollWindowDataset,
    TokenWindowDataset,
    build_corpus,
    build_tokenizer,
    split_scores,
)
from ..data.midi_io import Score
from ..models import build_model
from .trainer import Trainer, load_history, resolve_device

__all__ = ["Trainer", "load_history", "make_dataset", "resolve_device", "run_training", "set_seed"]

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds third-party code using the legacy API
    torch.manual_seed(seed)


def make_dataset(cfg: ExperimentConfig, scores: list[Score], train: bool) -> Dataset:
    d = cfg.data
    transpose = d.transpose_range if train else 0
    if d.representation == "tokens":
        return TokenWindowDataset(scores, build_tokenizer(d), d.seq_len, d.stride, transpose)
    return PianoRollWindowDataset(scores, d.min_pitch, d.max_pitch, d.seq_len, d.stride, transpose)


def run_training(cfg: ExperimentConfig, progress: bool = True) -> Trainer:
    set_seed(cfg.data.seed)
    corpus = build_corpus(cfg.data, progress=progress)
    splits = split_scores(corpus.scores, cfg.data.val_fraction, cfg.data.test_fraction,
                          cfg.data.seed)
    logger.info("Corpus: %s | split train/val/test = %d/%d/%d files", corpus.summary(),
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "splits.json", "w") as fh:
        json.dump({k: [s.name for s in v] for k, v in splits.items()}
                  | {"skipped": corpus.failures}, fh, indent=2)

    train_ds = make_dataset(cfg, splits["train"], train=True)
    val_ds = make_dataset(cfg, splits["val"], train=False) if splits["val"] else None
    if len(train_ds) == 0:
        raise ValueError("Training split produced no windows; add more data or lower seq_len")
    logger.info("Windows: %d train / %d val", len(train_ds), len(val_ds) if val_ds else 0)

    loader_kwargs = {"batch_size": cfg.train.batch_size, "num_workers": cfg.train.num_workers,
                     "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) >
                              cfg.train.batch_size, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds else None

    tokenizer = build_tokenizer(cfg.data)
    model = build_model(cfg, tokenizer)
    trainer = Trainer(model, cfg, tokenizer, train_loader, val_loader)
    trainer.fit()
    return trainer
