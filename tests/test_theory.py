import pytest

from musegen.theory import Key, Note, estimate_key, pitch_name


def scale_notes(tonic_pitch, steps, repeats=4):
    notes, t = [], 0
    for _ in range(repeats):
        for s in steps:
            notes.append(Note(t, tonic_pitch + s, 4))
            t += 4
        notes.append(Note(t, tonic_pitch, 8))  # cadence on the tonic
        t += 8
    return notes


@pytest.mark.parametrize(
    "text,tonic,mode",
    [("C major", 0, "major"), ("A minor", 9, "minor"), ("f# min", 6, "minor"), ("Bb", 10, "major"),
     ("Am", 9, "minor"), ("Eb major", 3, "major")],
)
def test_key_parse(text, tonic, mode):
    key = Key.parse(text)
    assert (key.tonic, key.mode) == (tonic, mode)


def test_key_parse_rejects_garbage():
    with pytest.raises(ValueError):
        Key.parse("H major")


def test_estimate_key_major_and_minor():
    assert estimate_key(scale_notes(60, [0, 2, 4, 5, 7, 9, 11, 7, 4])).name == "C major"
    assert estimate_key(scale_notes(57, [0, 2, 3, 5, 7, 8, 11, 7, 3])).name == "A minor"


def test_estimate_key_empty():
    assert estimate_key([]) is None


def test_key_contains_harmonic_minor_leading_tone():
    a_minor = Key.parse("A minor")
    assert a_minor.contains(68)  # G#
    assert not a_minor.contains(70)  # A#


def test_pitch_name():
    assert pitch_name(60) == "C4"
    assert pitch_name(69) == "A4"
