"""Export a trained Transformer and some held-out prompts for the in-browser demo.

The page in ``docs/`` runs the model in plain JavaScript (``docs/js/model.js``), so the
export is deliberately simple: one little-endian float16 blob with every tensor laid end
to end, and a JSON manifest giving each tensor's name, shape and offset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data.dataset import build_corpus, split_scores
from .models import count_parameters, load_checkpoint

FORMAT = "musegen-web-1"


def export_model(checkpoint: str | Path, out_dir: str | Path) -> dict:
    """Write ``model.bin`` and ``model.json`` to ``out_dir``; returns the manifest."""
    model, cfg, tokenizer, extra = load_checkpoint(checkpoint)
    if cfg.model.arch != "transformer":
        raise ValueError(f"Only the transformer runs in the browser, not {cfg.model.arch!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors, chunks, offset = [], [], 0
    for name, tensor in model.state_dict().items():
        if name == "head.weight" and cfg.model.tie_embeddings:
            continue  # tied to embed.weight
        data = tensor.detach().cpu().numpy().astype("<f2").ravel()
        tensors.append({"name": name, "shape": list(tensor.shape), "offset": offset,
                        "length": int(data.size)})
        chunks.append(data)
        offset += int(data.size)
    (out_dir / "model.bin").write_bytes(np.concatenate(chunks).tobytes())

    metrics = extra.get("metrics", {})
    manifest = {
        "format": FORMAT,
        "dtype": "float16",
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "d_model": cfg.model.d_model,
            "n_layers": cfg.model.n_layers,
            "n_heads": cfg.model.n_heads,
            "ff_mult": cfg.model.ff_mult,
            "max_len": cfg.model.max_len,
            "tie_embeddings": cfg.model.tie_embeddings,
        },
        "tokenizer": tokenizer.to_dict(),
        "training": {
            "n_params": count_parameters(model),
            "epoch": extra.get("epoch"),
            "val_loss": metrics.get("val_loss"),
            "val_perplexity": metrics.get("val_perplexity"),
            "seq_len": cfg.data.seq_len,
            "transpose_range": cfg.data.transpose_range,
        },
        "tensors": tensors,
    }
    (out_dir / "model.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def export_prompts(checkpoint: str | Path, out_path: str | Path, split: str = "test",
                   midi_dir: str | None = None) -> dict:
    """Write the held-out pieces of ``split`` as JSON note lists ``[start, pitch, dur, vel]``."""
    _, cfg, _, _ = load_checkpoint(checkpoint)
    corpus = build_corpus(cfg.data, midi_dir=midi_dir, progress=False)
    pieces = split_scores(corpus.scores, cfg.data.val_fraction, cfg.data.test_fraction,
                          cfg.data.seed)[split]
    payload = {
        "split": split,
        "steps_per_bar": cfg.data.steps_per_beat * cfg.data.beats_per_bar,
        "pieces": [
            {"name": s.name, "tempo": s.tempo, "bars": s.n_bars,
             "notes": [[n.start, n.pitch, n.duration, n.velocity] for n in s.notes]}
            for s in sorted(pieces, key=lambda s: s.name)
        ],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


def load_exported_weights(out_dir: str | Path) -> dict[str, np.ndarray]:
    """Read ``model.bin`` back into float32 arrays (used by the tests)."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "model.json").read_text())
    blob = np.frombuffer((out_dir / "model.bin").read_bytes(), dtype="<f2")
    return {t["name"]: blob[t["offset"] : t["offset"] + t["length"]].astype(np.float32)
            .reshape(t["shape"]) for t in manifest["tensors"]}
