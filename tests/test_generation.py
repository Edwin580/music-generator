import dataclasses

import pytest
import torch

from musegen.config import ExperimentConfig, GenerationConfig
from musegen.data.dataset import build_tokenizer
from musegen.data.midi_io import Score
from musegen.generation import filter_logits, generate, resolve_key
from musegen.models import build_model
from musegen.theory import Key, Note


def test_filter_logits_top_k_and_top_p():
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    probs = filter_logits(logits, top_k=2)
    assert (probs[2:] == 0).all() and torch.isclose(probs.sum(), torch.tensor(1.0))
    probs = filter_logits(logits, top_p=0.5)
    assert probs[0] == 1.0  # top token alone has > 50% of the mass
    probs = filter_logits(logits, mask=torch.tensor([False, True, True, False, False]))
    assert probs[0] == 0 and probs[3] == 0
    assert filter_logits(logits, temperature=0).argmax() == 0


def test_temperature_sharpens_and_flattens():
    logits = torch.tensor([2.0, 1.0, 0.0])
    cold, hot = filter_logits(logits, 0.5), filter_logits(logits, 2.0)
    assert cold[0] > filter_logits(logits)[0] > hot[0]


def test_resolve_key():
    notes = [Note(i * 4, p, 4) for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72, 67, 60])]
    assert resolve_key("auto", notes).name == "C major"
    assert resolve_key("none") is None
    assert resolve_key("G major") == Key(7, "major")


@pytest.mark.parametrize("arch", ["transformer", "lstm", "pianoroll_lstm"])
@pytest.mark.parametrize("with_prompt", [False, True])
def test_generate_respects_length_key_and_monophony(arch, with_prompt):
    torch.manual_seed(0)
    rep = "pianoroll" if arch == "pianoroll_lstm" else "tokens"
    cfg = ExperimentConfig.from_dict({"data": {"representation": rep},
                                      "model": {"arch": arch, "d_model": 32, "n_layers": 1,
                                                "n_heads": 2}})
    tok = build_tokenizer(cfg.data)
    model = build_model(cfg, tok)
    if arch == "pianoroll_lstm":  # untrained model is nearly silent; make it busy
        torch.nn.init.constant_(model.head.bias, 1.0)
    prompt = Score([Note(0, 62, 4), Note(4, 65, 4), Note(8, 69, 8)], bars=2) if with_prompt \
        else None
    gen = dataclasses.replace(GenerationConfig(), bars=4, key="D minor")
    result = generate(model, tok, gen, prompt, monophonic=True, seed=1)
    prompt_bars = 2 if with_prompt else 0
    new = [n for n in result.score.notes if n.start >= prompt_bars * 16]
    assert result.score.n_bars == prompt_bars + 4
    assert new, "expected some generated notes"
    assert all(n.start < (prompt_bars + 4) * 16 for n in new)
    assert all(Key.parse("D minor").contains(n.pitch) for n in new)
    starts = [n.start for n in new]
    assert len(starts) == len(set(starts))


def test_generation_is_reproducible_with_seed():
    cfg = ExperimentConfig.from_dict({"model": {"d_model": 32, "n_layers": 1, "n_heads": 2}})
    tok = build_tokenizer(cfg.data)
    model = build_model(cfg, tok)
    gen = dataclasses.replace(GenerationConfig(), bars=2)
    a = generate(model, tok, gen, seed=5).tokens
    b = generate(model, tok, gen, seed=5).tokens
    assert a == b
