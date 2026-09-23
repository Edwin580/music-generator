"""Typed experiment configuration.

Every experiment is described by a single :class:`ExperimentConfig`, which can be
loaded from YAML and overridden from the command line with ``section.key=value``
pairs (e.g. ``--set train.epochs=5 model.d_model=128``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PRESET_DIR = Path(__file__).parent / "configs"
REPRESENTATIONS = ("tokens", "pianoroll")
ARCHITECTURES = ("lstm", "transformer", "pianoroll_lstm")


@dataclass
class DataConfig:
    """How MIDI files are turned into training examples."""

    midi_dir: str = "data/midi"
    cache_dir: str = "data/cache"
    representation: str = "tokens"  # "tokens" (REMI events) or "pianoroll" (multi-hot frames)
    steps_per_beat: int = 4  # grid resolution: 4 -> sixteenth notes in 4/4
    beats_per_bar: int = 4
    melody_only: bool = True  # skyline-reduce to a monophonic melody
    min_pitch: int = 21  # A0 (piano range)
    max_pitch: int = 108  # C8
    max_duration_steps: int = 32  # longer notes are clipped (2 bars at 16ths)
    velocity_bins: int = 8
    seq_len: int = 256  # tokens (or frames for the piano-roll model) per training window
    stride: int = 128  # hop between consecutive windows of the same piece
    val_fraction: float = 0.1  # split is done per *file*, not per window
    test_fraction: float = 0.1
    transpose_range: int = 5  # random +/- semitone augmentation during training
    seed: int = 42


@dataclass
class ModelConfig:
    arch: str = "transformer"  # "lstm", "transformer" or "pianoroll_lstm"
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    max_len: int = 1024  # attention window of the transformer
    tie_embeddings: bool = True


@dataclass
class TrainConfig:
    output_dir: str = "runs/default"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    patience: int = 5  # early stopping on validation loss
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    amp: bool = True  # mixed precision (CUDA only)
    num_workers: int = 0
    log_every: int = 50
    max_steps_per_epoch: int | None = None  # handy for smoke tests


@dataclass
class GenerationConfig:
    bars: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = 0.95
    key: str | None = None  # e.g. "C major", "A minor", "auto" (detect from prompt) or None
    tempo: float = 120.0
    program: int = 0  # General MIDI program (0 = Acoustic Grand Piano)
    max_polyphony: int = 1  # piano-roll model only


@dataclass
class ExperimentConfig:
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.data.representation not in REPRESENTATIONS:
            raise ValueError(f"data.representation must be one of {REPRESENTATIONS}")
        if self.model.arch not in ARCHITECTURES:
            raise ValueError(f"model.arch must be one of {ARCHITECTURES}")
        expected = "pianoroll" if self.model.arch == "pianoroll_lstm" else "tokens"
        if self.data.representation != expected:
            raise ValueError(
                f"model.arch={self.model.arch!r} requires data.representation={expected!r}"
            )
        if not 0 <= self.data.min_pitch < self.data.max_pitch <= 127:
            raise ValueError("pitch range must satisfy 0 <= min_pitch < max_pitch <= 127")
        if self.data.val_fraction + self.data.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be < 1")
        if self.model.arch == "transformer" and self.model.d_model % self.model.n_heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        sections = {
            "data": DataConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "generation": GenerationConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in (raw or {}).items():
            if key in sections:
                kwargs[key] = _build(sections[key], value or {})
            elif key == "name":
                kwargs[key] = value
            else:
                raise KeyError(f"Unknown config section: {key!r}")
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    @classmethod
    def load(cls, name_or_path: str | Path) -> ExperimentConfig:
        """Load a YAML file, or a bundled preset by name (``transformer``, ``lstm``, ...)."""
        path = Path(name_or_path)
        if path.exists():
            return cls.from_yaml(path)
        preset = PRESET_DIR / f"{path.stem}.yaml"
        if preset.exists():
            return cls.from_yaml(preset)
        raise FileNotFoundError(
            f"No config file {str(name_or_path)!r} and no preset of that name "
            f"(presets: {', '.join(list_presets())})"
        )

    def save_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def with_overrides(self, overrides: list[str] | None) -> ExperimentConfig:
        """Return a copy with ``section.key=value`` overrides applied (values parsed as YAML)."""
        raw = self.to_dict()
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"Override must look like section.key=value, got {item!r}")
            dotted, value = item.split("=", 1)
            parts = dotted.split(".")
            target = raw
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    raise KeyError(f"Unknown config path: {dotted!r}")
                target = target[part]
            if parts[-1] not in target:
                raise KeyError(f"Unknown config key: {dotted!r}")
            target[parts[-1]] = yaml.safe_load(value)
        return ExperimentConfig.from_dict(raw)


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def _build(klass: type, values: dict[str, Any]) -> Any:
    known = {f.name for f in fields(klass)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"Unknown keys for {klass.__name__}: {sorted(unknown)}")
    return klass(**values)


def replace(obj: Any, **changes: Any) -> Any:
    """Thin re-export of :func:`dataclasses.replace` for convenience."""
    return dataclasses.replace(obj, **changes)
