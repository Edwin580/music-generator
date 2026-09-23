import numpy as np
import pytest

from musegen.data.pianoroll import notes_to_roll, roll_to_notes, transpose_roll
from musegen.data.tokenizer import DUR, PITCH, GrammarState, REMITokenizer
from musegen.theory import Key, Note

TOK = REMITokenizer()
MELODY = [Note(0, 60, 4, 90), Note(4, 60, 2, 90), Note(6, 62, 2, 60), Note(16, 64, 12, 100),
          Note(40, 67, 32, 80)]


def strip_velocity(notes):
    return [(n.start, n.pitch, n.duration) for n in notes]


def test_token_round_trip():
    tokens = TOK.encode(MELODY)
    assert tokens[0] == TOK.bos_id and tokens[-1] == TOK.eos_id
    assert TOK.count_bars(tokens) == 3
    assert strip_velocity(TOK.decode(tokens)) == strip_velocity(MELODY)


def test_empty_bars_are_encoded():
    tokens = TOK.encode([Note(64, 60, 4)])  # first note in bar 5
    assert TOK.count_bars(tokens) == 5
    assert TOK.decode(tokens)[0].start == 64


def test_velocity_quantization_is_close():
    decoded = TOK.decode(TOK.encode(MELODY))
    for a, b in zip(decoded, MELODY, strict=True):
        assert abs(a.velocity - b.velocity) <= 127 / TOK.velocity_bins


def test_transpose_tokens():
    tokens = np.array(TOK.encode(MELODY))
    up = TOK.transpose(tokens, 5)
    assert [n.pitch for n in TOK.decode(up)] == [n.pitch + 5 for n in MELODY]
    assert TOK.transpose(np.array(TOK.encode([Note(0, 107, 1)])), 2) is None


def test_decode_tolerates_garbage():
    rng = np.random.default_rng(0)
    TOK.decode(rng.integers(0, TOK.vocab_size, 500).tolist())  # must not raise


@pytest.mark.parametrize("monophonic", [True, False])
def test_grammar_constrained_random_walk_is_well_formed(monophonic):
    rng = np.random.default_rng(1)
    key = Key.parse("E minor")
    for _ in range(20):
        state = GrammarState(TOK, key=key, monophonic=monophonic)
        seq = []
        for _ in range(120):
            allowed = np.flatnonzero(state.allowed(allow_eos=False))
            tok = int(rng.choice(allowed))
            seq.append(tok)
            state.update(tok)
        types = TOK.token_types()[seq]
        # every PITCH is followed by a DUR, and all pitches are in key
        for i in np.flatnonzero(types == PITCH):
            if i + 1 < len(seq):
                assert types[i + 1] == DUR
        notes = TOK.decode(seq)
        assert all(key.contains(n.pitch) for n in notes)
        # positions within a bar strictly increase, so onsets are sorted
        starts = [n.start for n in notes]
        assert starts == sorted(starts)
        if monophonic:
            assert len(starts) == len(set(starts))


def test_pianoroll_round_trip_distinguishes_repeats_from_holds():
    roll = notes_to_roll(MELODY, 21, 108)
    assert roll.shape == (72, 2 * 88)
    assert strip_velocity(roll_to_notes(roll, 21)) == strip_velocity(sorted(MELODY))


def test_pianoroll_transpose():
    roll = notes_to_roll(MELODY, 21, 108)
    up = transpose_roll(roll, 3)
    assert [n.pitch for n in roll_to_notes(up, 21)] == [n.pitch + 3 for n in sorted(MELODY)]
    top = notes_to_roll([Note(0, 108, 2)], 21, 108)
    assert transpose_roll(top, 1) is None
