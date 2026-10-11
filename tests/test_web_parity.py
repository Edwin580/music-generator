"""The browser demo (docs/js) must compute the same things as the Python package.

Each test sends inputs to tests/web/parity.mjs under Node and compares its answers with
the Python implementation. Skipped when Node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from musegen.config import ExperimentConfig
from musegen.data.dataset import build_tokenizer
from musegen.data.midi_io import Score
from musegen.data.sources import _synth_piece
from musegen.data.tokenizer import GrammarState, REMITokenizer
from musegen.evaluation.metrics import repetition_ratio, scale_consistency
from musegen.generation import filter_logits
from musegen.models import build_model, save_checkpoint
from musegen.theory import Key, Note, estimate_key, key_correlations
from musegen.web_export import export_model, export_prompts, load_exported_weights

NODE = shutil.which("node")
SCRIPT = Path(__file__).parent / "web" / "parity.mjs"
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def run_js(tmp_path: Path, request: dict) -> dict:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    done = subprocess.run([NODE, str(SCRIPT), str(path)], capture_output=True, text=True,
                          check=True, timeout=120)
    return json.loads(done.stdout)


def tiny_checkpoint(tmp_path: Path, max_len: int = 1024):
    cfg = ExperimentConfig.from_dict({"model": {"d_model": 32, "n_layers": 2, "n_heads": 4,
                                                "max_len": max_len}})
    tok = build_tokenizer(cfg.data)
    torch.manual_seed(0)
    model = build_model(cfg, tok).eval()
    with torch.no_grad():  # larger weights than the init, so the test is not trivially passed
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, cfg, tok)
    return path, model, tok


def piece(seed: int) -> list[Note]:
    from musegen.data.midi_io import quantize_midi

    return quantize_midi(_synth_piece(np.random.default_rng(seed))).notes


def rows(notes):
    return [[n.start, n.pitch, n.duration, n.velocity] for n in notes]


@pytest.mark.parametrize("max_len", [1024, 16])
def test_js_transformer_matches_pytorch(tmp_path, max_len):
    ckpt, model, tok = tiny_checkpoint(tmp_path, max_len)
    export_model(ckpt, tmp_path / "web")
    # PyTorch with the same float16-rounded weights the browser gets
    weights = load_exported_weights(tmp_path / "web")
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()}, strict=False)
    tokens = tok.encode(piece(3))[:40]

    js = np.array(run_js(tmp_path, {"model": {"dir": str(tmp_path / "web"),
                                              "tokens": tokens}})["logits"])
    with torch.no_grad():
        if max_len >= len(tokens):
            expected = model(torch.tensor([tokens]))[0][0].numpy()
        else:  # step through the sliding KV cache, as generation does
            state, steps = None, []
            for t in tokens:
                logits, state = model(torch.tensor([[t]]), state)
                steps.append(logits[0, 0].numpy())
            expected = np.stack(steps)
    assert js.shape == expected.shape
    np.testing.assert_allclose(js, expected, atol=2e-4, rtol=1e-4)


def test_export_round_trip_and_prompts(tmp_path, tiny_config):
    cfg = tiny_config()
    tok = build_tokenizer(cfg.data)
    model = build_model(cfg, tok)
    save_checkpoint(tmp_path / "c.pt", model, cfg, tok)
    manifest = export_model(tmp_path / "c.pt", tmp_path / "web")
    weights = load_exported_weights(tmp_path / "web")
    assert "head.weight" not in weights  # tied to the embedding
    for name, tensor in model.state_dict().items():
        if name != "head.weight":
            np.testing.assert_allclose(weights[name], tensor.numpy(), atol=1e-3, rtol=1e-3)
    assert manifest["config"]["vocab_size"] == tok.vocab_size

    prompts = export_prompts(tmp_path / "c.pt", tmp_path / "web" / "prompts.json")
    assert prompts["pieces"] and all(p["notes"] for p in prompts["pieces"])

    cfg_lstm = ExperimentConfig.from_dict({"model": {"arch": "lstm", "d_model": 16}})
    save_checkpoint(tmp_path / "l.pt", build_model(cfg_lstm, tok), cfg_lstm, tok)
    with pytest.raises(ValueError):
        export_model(tmp_path / "l.pt", tmp_path / "web2")


@pytest.mark.parametrize("key,monophonic", [(None, True), ("E minor", True), ("Bb major", False)])
def test_js_tokenizer_and_grammar_match_python(tmp_path, key, monophonic):
    tok = REMITokenizer()
    notes = piece(5)
    tokens = tok.encode(notes)
    rng = np.random.default_rng(2)
    noisy = tokens[:30] + rng.integers(0, tok.vocab_size, 40).tolist()  # garbage must decode alike
    # a grammatical random walk under the constraint
    state = GrammarState(tok, key=Key.parse(key) if key else None, monophonic=monophonic)
    walk, masks = [], []
    for _ in range(150):
        mask = state.allowed(allow_eos=True)
        masks.append(mask.astype(int).tolist())
        choices = np.flatnonzero(state.allowed(allow_eos=False))
        t = int(rng.choice(choices))
        walk.append(t)
        state.update(t)

    js = run_js(tmp_path, {"tokenizer": {"config": tok.to_dict(), "notes": rows(notes),
                                         "tokens": noisy, "key": key, "monophonic": monophonic,
                                         "walk": walk}})
    assert js["vocabSize"] == tok.vocab_size
    assert js["encoded"] == tokens
    assert js["decoded"] == rows(tok.decode(noisy))
    assert js["masks"] == masks


def test_js_sampling_filter_matches_python(tmp_path):
    rng = np.random.default_rng(0)
    cases = []
    for temperature, top_k, top_p in [(1.0, None, None), (0.7, None, 0.9), (1.5, 5, 0.5),
                                      (0.0, None, 0.9), (1.0, None, 0.95)]:
        logits = rng.normal(0, 3, 60)
        mask = rng.random(60) < 0.6
        cases.append({"logits": logits.tolist(), "mask": mask.astype(int).tolist(),
                      "temperature": temperature, "top_k": top_k, "top_p": top_p})
    js = run_js(tmp_path, {"filter": {"cases": cases}})["filtered"]
    for case, got in zip(cases, js, strict=True):
        expected = filter_logits(torch.tensor(case["logits"]), case["temperature"],
                                 case["top_k"], case["top_p"],
                                 torch.tensor(case["mask"], dtype=torch.bool))
        np.testing.assert_allclose(got, expected.numpy(), atol=1e-6)


def test_js_key_finding_and_metrics_match_python(tmp_path):
    pieces = [piece(s) for s in range(6)]
    hists = [np.random.default_rng(s).integers(0, 9, 12).tolist() for s in range(5)] + [[0] * 12]
    js = run_js(tmp_path, {"theory": {"histograms": hists,
                                      "pieces": [rows(p) for p in pieces]}})
    for h, got in zip(hists, js["correlations"], strict=True):
        np.testing.assert_allclose(got, key_correlations(h), atol=1e-9)
    for notes, got in zip(pieces, js["metrics"], strict=True):
        score = Score(notes)
        assert got["key"] == estimate_key(notes).name
        assert got["scale_consistency"] == pytest.approx(scale_consistency(score))
        assert got["key_clarity"] == pytest.approx(key_correlations(
            np.bincount([n.pitch % 12 for n in notes], weights=[n.duration for n in notes],
                        minlength=12)).max())
        assert got["repetition_ratio"] == pytest.approx(repetition_ratio(score))
