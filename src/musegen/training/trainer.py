"""Training loop.

Replaces ``model.fit(..., metrics=['accuracy'])`` from the notebook. Keras "accuracy" on
a 128-way multi-hot sigmoid output (mostly zeros) is hard to interpret; the 25%
"accuracy" in the original log isn't a meaningful number. We report:

* token models: cross-entropy, **perplexity** and next-token accuracy (ignoring padding);
* piano-roll model: BCE plus **frame-level precision / recall / F1** of the onset and
  sustain predictions.

Also included: AdamW with decoupled weight decay, linear warm-up plus cosine decay,
gradient clipping, mixed precision on CUDA, early stopping, best/last checkpoints,
and a CSV training history.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.tokenizer import REMITokenizer
from ..models import count_parameters, save_checkpoint

logger = logging.getLogger(__name__)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warmup_cosine(step: int, warmup: int, total: int, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def loss_and_stats(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                   pad_id: int = 0) -> tuple[torch.Tensor, dict]:
    """Loss for one batch plus counts needed for epoch-level metrics."""
    logits, _ = model(x)
    if getattr(model, "kind", "tokens") == "tokens":
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=pad_id
        )
        valid = y != pad_id
        correct = ((logits.argmax(-1) == y) & valid).sum()
        return loss, {"n": valid.sum().item(), "correct": correct.item()}

    loss = F.binary_cross_entropy_with_logits(logits.float(), y)
    pred = logits > 0
    truth = y > 0.5
    return loss, {
        "n": y.numel(),
        "tp": (pred & truth).sum().item(),
        "fp": (pred & ~truth).sum().item(),
        "fn": (~pred & truth).sum().item(),
    }


def summarise(total_loss: float, batches: int, stats: dict, kind: str) -> dict:
    loss = total_loss / max(1, batches)
    out = {"loss": loss}
    if kind == "tokens":
        out["perplexity"] = math.exp(min(loss, 50))
        out["accuracy"] = stats.get("correct", 0) / max(1, stats.get("n", 0))
    else:
        tp, fp, fn = stats.get("tp", 0), stats.get("fp", 0), stats.get("fn", 0)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        out.update(precision=precision, recall=recall,
                   f1=2 * precision * recall / max(1e-9, precision + recall))
    return out


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device,
                    pad_id: int = 0) -> dict:
    model.eval()
    total, batches, stats = 0.0, 0, {}
    for x, y in loader:
        loss, batch_stats = loss_and_stats(model, x.to(device), y.to(device), pad_id)
        total += loss.item()
        batches += 1
        for k, v in batch_stats.items():
            stats[k] = stats.get(k, 0) + v
    return summarise(total, batches, stats, getattr(model, "kind", "tokens"))


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        tokenizer: REMITokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.tc = cfg.train
        self.tokenizer = tokenizer
        self.device = resolve_device(self.tc.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.kind = getattr(model, "kind", "tokens")
        self.out_dir = Path(self.tc.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 or "embed" in name or "ln" in name else decay).append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.tc.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.tc.lr, betas=(0.9, 0.98),
        )
        steps_per_epoch = len(train_loader)
        if self.tc.max_steps_per_epoch:
            steps_per_epoch = min(steps_per_epoch, self.tc.max_steps_per_epoch)
        total = max(1, steps_per_epoch * self.tc.epochs)
        # On small datasets a fixed warm-up can exceed the whole run; cap it at 10%.
        warmup = min(self.tc.warmup_steps, max(1, total // 10))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: warmup_cosine(s, warmup, total)
        )
        self.use_amp = self.tc.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: list[dict] = []
        self.global_step = 0

    # --------------------------------------------------------------- epochs
    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        total_loss, batches, stats = 0.0, 0, {}
        limit = self.tc.max_steps_per_epoch if train else None
        for step, (x, y) in enumerate(loader):
            if limit and step >= limit:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.set_grad_enabled(train), torch.autocast(
                self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                loss, batch_stats = loss_and_stats(self.model, x, y,
                                                     self.tokenizer.pad_id)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1
                if self.global_step % self.tc.log_every == 0:
                    logger.info("step %d  loss %.4f  lr %.2e", self.global_step, loss.item(),
                                self.scheduler.get_last_lr()[0])
            total_loss += loss.item()
            batches += 1
            for k, v in batch_stats.items():
                stats[k] = stats.get(k, 0) + v
        return summarise(total_loss, batches, stats, self.kind)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        return self._run_epoch(loader, train=False)

    def fit(self) -> list[dict]:
        n_params = count_parameters(self.model)
        logger.info("Training %s (%s parameters) on %s", self.cfg.model.arch,
                    f"{n_params:,}", self.device)
        self.cfg.save_yaml(self.out_dir / "config.yaml")
        best, bad_epochs = float("inf"), 0
        for epoch in range(1, self.tc.epochs + 1):
            start = time.time()
            train_metrics = self._run_epoch(self.train_loader, train=True)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
            if self.val_loader is not None and len(self.val_loader):
                row.update({f"val_{k}": v for k, v in self.evaluate(self.val_loader).items()})
            row["lr"] = self.scheduler.get_last_lr()[0]
            row["seconds"] = round(time.time() - start, 2)
            self.history.append(row)
            self._write_history()
            logger.info("epoch %d  %s", epoch, "  ".join(
                f"{k}={v:.4g}" for k, v in row.items() if isinstance(v, float)))

            monitored = row.get("val_loss", row["train_loss"])
            extra = {"epoch": epoch, "metrics": row, "n_params": n_params}
            save_checkpoint(self.out_dir / "last.pt", self.model, self.cfg, self.tokenizer, extra)
            if monitored < best - 1e-4:
                best, bad_epochs = monitored, 0
                save_checkpoint(self.out_dir / "best.pt", self.model, self.cfg, self.tokenizer,
                                extra)
            else:
                bad_epochs += 1
                if bad_epochs >= self.tc.patience:
                    logger.info("Early stopping after %d epochs without improvement", bad_epochs)
                    break
        with open(self.out_dir / "summary.json", "w") as fh:
            json.dump({"best_monitored_loss": best, "epochs_run": len(self.history),
                       "n_params": n_params, "final": self.history[-1]}, fh, indent=2)
        return self.history

    def _write_history(self) -> None:
        keys = sorted({k for row in self.history for k in row}, key=lambda k: (k != "epoch", k))
        with open(self.out_dir / "history.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)


def load_history(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [{k: float(v) if v not in ("", None) else None for k, v in row.items()}
                for row in csv.DictReader(fh)]
