import numpy as np

from musegen.data.midi_io import Score
from musegen.evaluation.metrics import (
    aggregate,
    compare_sets,
    js_divergence,
    overlap_area,
    piece_metrics,
    repetition_ratio,
    scale_consistency,
)
from musegen.theory import Note

MOTIF = [(0, 60, 4), (4, 62, 2), (6, 64, 2), (8, 67, 8)]


def motif_score(repeats=4, transpose_each=0):
    notes = []
    for r in range(repeats):
        notes += [Note(16 * r + s, p + r * transpose_each, d) for s, p, d in MOTIF]
    return Score(notes)


def test_scale_consistency():
    assert scale_consistency(motif_score()) == 1.0
    chromatic = Score([Note(i * 2, 60 + i, 2) for i in range(12)])
    assert scale_consistency(chromatic) < 1.0


def test_repetition_is_transposition_invariant():
    assert repetition_ratio(motif_score(transpose_each=0)) > 0.5
    assert repetition_ratio(motif_score(transpose_each=2)) > 0.3
    rng = np.random.default_rng(0)
    random_line = Score([Note(i * 2, int(p), int(d)) for i, (p, d) in
                         enumerate(zip(rng.integers(48, 84, 64), rng.integers(1, 8, 64),
                                       strict=True))])
    assert repetition_ratio(random_line) < 0.1


def test_piece_metrics_and_aggregate():
    m = piece_metrics(motif_score())
    assert m["n_notes"] == 16 and m["n_bars"] == 4 and m["notes_per_bar"] == 4
    assert m["empty_bar_rate"] == 0 and m["polyphony_rate"] == 0
    agg = aggregate([motif_score(), motif_score(2)])
    assert agg["n_notes"]["mean"] == 12


def test_distribution_similarity():
    p = np.array([1.0, 2.0, 3.0])
    assert np.isclose(overlap_area(p, p), 1.0) and np.isclose(js_divergence(p, p), 0.0)
    q = np.array([3.0, 0.0, 0.0])
    assert js_divergence(p, q) > 0.3 and overlap_area(p, q) < 0.5
    same = compare_sets([motif_score()], [motif_score()])
    assert all(np.isclose(v["overlap_area"], 1.0) for v in same.values())
