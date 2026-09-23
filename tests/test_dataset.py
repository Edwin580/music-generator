import torch

from musegen.data.dataset import (
    PianoRollWindowDataset,
    TokenWindowDataset,
    _window_starts,
    build_corpus,
    build_tokenizer,
    split_scores,
)


def test_corpus_and_cache(tiny_config):
    cfg = tiny_config()
    corpus = build_corpus(cfg.data, progress=False)
    assert len(corpus.scores) == 12
    assert all(len(s.notes) > 8 for s in corpus.scores)
    again = build_corpus(cfg.data, progress=False)  # served from cache
    assert [s.name for s in again.scores] == [s.name for s in corpus.scores]


def test_split_is_by_file_disjoint_and_deterministic(tiny_config):
    cfg = tiny_config()
    scores = build_corpus(cfg.data, progress=False).scores
    a = split_scores(scores, 0.2, 0.2, seed=3)
    b = split_scores(list(reversed(scores)), 0.2, 0.2, seed=3)
    names = {k: {s.name for s in v} for k, v in a.items()}
    assert names == {k: {s.name for s in v} for k, v in b.items()}
    assert not names["train"] & names["val"] and not names["train"] & names["test"]
    assert sum(len(v) for v in names.values()) == len(scores)
    assert names["val"] and names["test"]


def test_window_starts_cover_sequence():
    starts = _window_starts(100, 30, 20)
    assert starts[0] == 0 and starts[-1] == 70
    assert _window_starts(10, 30, 20) == [0]
    assert _window_starts(1, 30, 20) == []


def test_token_windows_are_bar_aligned(tiny_config):
    cfg = tiny_config()
    scores = build_corpus(cfg.data, progress=False).scores
    tok = build_tokenizer(cfg.data)
    ds = TokenWindowDataset(scores, tok, seq_len=64, stride=32)
    # every window except each piece's final (end-flush) one starts on a bar line
    last_start = {}
    for i, s in ds.windows:
        last_start[i] = max(s, last_start.get(i, 0))
    first_tokens = {ds.sequences[i][s] for i, s in ds.windows if 0 < s < last_start[i]}
    assert first_tokens
    assert first_tokens <= {tok.bar_id}
    x, y = ds[0]
    assert x.shape == y.shape == (64,) and x.dtype == torch.long
    assert torch.equal(x[1:], y[:-1])


def test_transposition_augmentation_changes_pitches_only(tiny_config):
    cfg = tiny_config()
    scores = build_corpus(cfg.data, progress=False).scores
    tok = build_tokenizer(cfg.data)
    ds = TokenWindowDataset(scores[:2], tok, 64, 32, transpose_range=6)
    torch.manual_seed(0)
    types = torch.from_numpy(tok.token_types())
    x0 = torch.from_numpy(ds.sequences[0][:64])
    x, _ = ds[0]
    assert torch.equal(types[x], types[x0])


def test_pianoroll_windows(tiny_config):
    cfg = tiny_config("pianoroll_lstm")
    scores = build_corpus(cfg.data, progress=False).scores
    ds = PianoRollWindowDataset(scores, 21, 108, seq_len=32, stride=16, transpose_range=2)
    x, y = ds[0]
    assert x.shape == y.shape == (32, 176)
