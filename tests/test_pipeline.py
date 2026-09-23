"""End-to-end: train each architecture briefly, then generate, evaluate and render via the CLI."""

import json
import wave

import pytest

from musegen.cli import main
from musegen.training import run_training


@pytest.mark.parametrize("arch", ["transformer", "lstm", "pianoroll_lstm"])
def test_train_generate_evaluate(arch, tiny_config, tmp_path):
    cfg = tiny_config(arch)
    trainer = run_training(cfg, progress=False)
    run = tmp_path / "run"
    assert (run / "best.pt").exists() and (run / "history.csv").exists()
    assert trainer.history[0]["train_loss"] > 0
    splits = json.loads((run / "splits.json").read_text())
    assert not set(splits["train"]) & set(splits["val"])

    out = tmp_path / "gen" / "song.mid"
    assert main(["generate", "-m", str(run / "best.pt"), "-o", str(out), "--bars", "2",
                 "--seed", "0", "--plot", "--wav", "--key", "C major"]) == 0
    assert out.exists() and out.with_suffix(".png").exists()
    with wave.open(str(out.with_suffix(".wav"))) as wf:
        assert wf.getsampwidth() == 2 and wf.getnframes() > 0

    eval_dir = tmp_path / "eval"
    assert main(["evaluate", "-m", str(run / "best.pt"), "--out-dir", str(eval_dir),
                 "--num-samples", "2", "--bars", "2", "--prompt-bars", "1"]) == 0
    report = json.loads((eval_dir / "report.json").read_text())
    assert report["held_out_likelihood"]["loss"] > 0
    assert (eval_dir / "report.md").exists()


def test_cli_analyze_and_render(synthetic_dir, tmp_path, capsys):
    midi = sorted(synthetic_dir.glob("*.mid"))[0]
    assert main(["analyze", str(midi), "--plot", str(tmp_path / "a.png")]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["key"] and info["scale_consistency"] > 0.9
    assert (tmp_path / "a_roll.png").exists() and (tmp_path / "a_key.png").exists()
    assert main(["render", str(midi), "--wav", str(tmp_path / "a.wav")]) == 0


def test_cli_reports_missing_data(tmp_path):
    assert main(["train", "-c", "smoke", "--midi-dir", str(tmp_path / "nothing"),
                 "--output-dir", str(tmp_path / "r")]) == 2
