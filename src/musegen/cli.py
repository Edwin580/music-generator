"""Command-line interface: ``musegen <command> ...``.

Commands
--------
download    fetch the MIDIWorld sample set used by the original notebook
synth-data  write a procedurally composed corpus (offline demo / testing)
train       train a model from a YAML config
generate    sample new music (optionally continuing a MIDI prompt)
evaluate    held-out likelihood + musical statistics + figures
analyze     key, statistics and piano roll for any MIDI file
render      MIDI -> WAV
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("musegen")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------ commands
def cmd_download(args) -> int:
    from .data.sources import download_midiworld

    files = download_midiworld(args.dest)
    print(f"{len(files)} MIDI files in {args.dest}")
    return 0 if files else 1


def cmd_synth(args) -> int:
    from .data.sources import generate_synthetic_corpus

    files = generate_synthetic_corpus(args.dest, args.n, args.seed)
    print(f"Wrote {len(files)} synthetic pieces to {args.dest}")
    return 0


def cmd_train(args) -> int:
    from .config import ExperimentConfig
    from .training import run_training

    cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    overrides = list(args.set or [])
    if args.midi_dir:
        overrides.append(f"data.midi_dir={args.midi_dir}")
    if args.output_dir:
        overrides.append(f"train.output_dir={args.output_dir}")
    cfg = cfg.with_overrides(overrides)
    trainer = run_training(cfg)
    best = Path(cfg.train.output_dir) / "best.pt"
    print(f"Done. Best checkpoint: {best}")
    if trainer.history:
        from .visualization import plot_training_history

        plot_training_history(trainer.history, Path(cfg.train.output_dir) / "training_history.png")
    return 0


def _generation_config(args, base):
    changes = {k: getattr(args, k) for k in ("bars", "temperature", "top_k", "top_p", "key",
                                             "tempo", "program") if getattr(args, k) is not None}
    return dataclasses.replace(base, **changes)


def cmd_generate(args) -> int:
    from .audio import render, write_wav
    from .data.midi_io import load_score, notes_to_midi, save_score
    from .generation import generate
    from .models import load_checkpoint
    from .training import resolve_device
    from .visualization import plot_piano_roll

    model, cfg, tokenizer, _ = load_checkpoint(args.checkpoint, resolve_device(args.device))
    gen = _generation_config(args, cfg.generation)

    prompt = None
    if args.prompt:
        d = cfg.data
        full = load_score(args.prompt, steps_per_beat=d.steps_per_beat,
                          beats_per_bar=d.beats_per_bar, melody_only=d.melody_only,
                          min_pitch=d.min_pitch, max_pitch=d.max_pitch,
                          max_duration=d.max_duration_steps)
        prompt = full.slice_bars(0, args.prompt_bars)
        if args.tempo is None:
            gen = dataclasses.replace(gen, tempo=full.tempo)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    for i in range(args.num):
        seed = None if args.seed is None else args.seed + i
        result = generate(model, tokenizer, gen, prompt, monophonic=cfg.data.melody_only,
                          seed=seed)
        path = out if args.num == 1 else Path(f"{stem}_{i:02d}.mid")
        save_score(result.score, path, gen.program)
        key = f" in {result.key}" if result.key else ""
        print(f"Wrote {path} ({len(result.score.notes)} notes, {result.score.n_bars} bars{key})")
        if args.plot:
            plot_piano_roll(result.score, path.with_suffix(".png"), title=path.stem,
                            prompt_bars=result.prompt_bars)
        if args.wav:
            pm = notes_to_midi(result.score.notes, result.score.steps_per_beat, gen.tempo,
                               gen.program)
            write_wav(render(pm, soundfont=args.soundfont), path.with_suffix(".wav"))
    return 0


def cmd_evaluate(args) -> int:
    from .evaluation import evaluate_checkpoint

    out_dir = args.out_dir or str(Path(args.checkpoint).parent / "eval")
    report = evaluate_checkpoint(args.checkpoint, out_dir, n_samples=args.num_samples,
                                 bars=args.bars, prompt_bars=args.prompt_bars,
                                 device=args.device, seed=args.seed)
    print((Path(out_dir) / "report.md").read_text())
    print(f"Figures and samples written to {out_dir}")
    return 0 if report else 1


def cmd_analyze(args) -> int:
    from .data.midi_io import load_score
    from .evaluation.metrics import piece_metrics
    from .theory import estimate_key

    score = load_score(args.midi, melody_only=not args.polyphonic)
    key = estimate_key(score.notes)
    info = {"file": args.midi, "tempo": score.tempo, "bars": score.n_bars,
            "key": key.name if key else None,
            "key_confidence": round(key.confidence, 3) if key else None,
            **{k: round(v, 4) for k, v in piece_metrics(score).items()}}
    print(json.dumps(info, indent=2))
    if args.plot:
        from .visualization import plot_key_profile, plot_piano_roll

        base = Path(args.plot)
        plot_piano_roll(score, base.with_name(base.stem + "_roll.png"), title=score.name,
                        max_bars=args.max_bars)
        plot_key_profile(score, base.with_name(base.stem + "_key.png"))
    return 0


def cmd_render(args) -> int:
    from .audio import midi_to_wav

    path = midi_to_wav(args.midi, args.wav or Path(args.midi).with_suffix(".wav"),
                       args.sample_rate, args.soundfont)
    print(f"Wrote {path}")
    return 0


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="musegen", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download the notebook's MIDIWorld sample set")
    p.add_argument("--dest", default="data/midi")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("synth-data", help="write a synthetic corpus for offline experiments")
    p.add_argument("--dest", default="data/synthetic")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("train", help="train a model")
    p.add_argument("--config", "-c", help="YAML file or preset name: transformer, lstm, pianoroll_lstm, smoke")
    p.add_argument("--midi-dir")
    p.add_argument("--output-dir")
    p.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                   help="config overrides, e.g. train.epochs=5 model.d_model=128")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("generate", help="generate music from a checkpoint")
    p.add_argument("--checkpoint", "-m", required=True)
    p.add_argument("--out", "-o", default="output/generated.mid")
    p.add_argument("--num", "-n", type=int, default=1)
    p.add_argument("--bars", type=int)
    p.add_argument("--temperature", "-t", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--top-p", type=float)
    p.add_argument("--key", help="'C major', 'A minor', 'auto' (from prompt) or 'none'")
    p.add_argument("--tempo", type=float)
    p.add_argument("--program", type=int, help="General MIDI program number")
    p.add_argument("--prompt", help="MIDI file whose opening bars seed the generation")
    p.add_argument("--prompt-bars", type=int, default=4)
    p.add_argument("--seed", type=int)
    p.add_argument("--plot", action="store_true", help="also save a piano-roll PNG")
    p.add_argument("--wav", action="store_true", help="also render a WAV file")
    p.add_argument("--soundfont", help="SoundFont (.sf2) for FluidSynth rendering")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p.add_argument("--checkpoint", "-m", required=True)
    p.add_argument("--out-dir")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--bars", type=int, default=16)
    p.add_argument("--prompt-bars", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("analyze", help="analyze a MIDI file")
    p.add_argument("midi")
    p.add_argument("--polyphonic", action="store_true", help="analyze all parts, not the melody")
    p.add_argument("--plot", metavar="PNG", help="write piano-roll and key-profile figures")
    p.add_argument("--max-bars", type=int, default=32)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("render", help="render a MIDI file to WAV")
    p.add_argument("midi")
    p.add_argument("--wav")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--soundfont")
    p.set_defaults(func=cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
