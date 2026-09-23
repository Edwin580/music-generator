import pytest
import torch

from musegen.config import ExperimentConfig
from musegen.data.dataset import build_tokenizer
from musegen.models import build_model, load_checkpoint, save_checkpoint
from musegen.models.lstm import MusicLSTM
from musegen.models.transformer import MusicTransformer

V = 64


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("model", [
    MusicTransformer(V, d_model=32, n_layers=2, n_heads=4, dropout=0.0, max_len=64),
    MusicLSTM(V, d_model=32, n_layers=2, dropout=0.0),
])
def test_incremental_decoding_matches_full_forward(model):
    model.eval()
    x = torch.randint(1, V, (2, 20))
    full, _ = model(x)
    logits, state = model(x[:, :7])
    steps = [logits]
    for t in range(7, 20):
        logits, state = model(x[:, t : t + 1], state)
        steps.append(logits)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_transformer_is_causal():
    model = MusicTransformer(V, d_model=32, n_layers=2, n_heads=4, dropout=0.0).eval()
    x = torch.randint(1, V, (1, 16))
    y = x.clone()
    y[0, 10:] = torch.randint(1, V, (6,))
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_transformer_sliding_kv_cache_runs_past_max_len():
    model = MusicTransformer(V, d_model=32, n_layers=1, n_heads=4, dropout=0.0, max_len=8).eval()
    logits, state = model(torch.randint(1, V, (1, 8)))
    for _ in range(20):
        logits, state = model(torch.randint(1, V, (1, 1)), state)
    assert state["cache"][0][0].shape[2] == 8 and state["pos"] == 28
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("arch", ["transformer", "lstm", "pianoroll_lstm"])
def test_build_and_checkpoint_round_trip(arch, tmp_path):
    rep = "pianoroll" if arch == "pianoroll_lstm" else "tokens"
    cfg = ExperimentConfig.from_dict({"data": {"representation": rep},
                                      "model": {"arch": arch, "d_model": 32, "n_layers": 1,
                                                "n_heads": 2}})
    tok = build_tokenizer(cfg.data)
    model = build_model(cfg, tok).eval()
    save_checkpoint(tmp_path / "m.pt", model, cfg, tok, {"epoch": 3})
    loaded, cfg2, tok2, extra = load_checkpoint(tmp_path / "m.pt")
    assert cfg2 == cfg and tok2 == tok and extra["epoch"] == 3
    x = (torch.randint(1, tok.vocab_size, (1, 5)) if rep == "tokens"
         else torch.rand(1, 5, 2 * tok.n_pitches).round())
    assert torch.allclose(model(x)[0], loaded(x)[0])
