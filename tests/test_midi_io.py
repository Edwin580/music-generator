import mido
import pretty_midi

from musegen.data.midi_io import (
    load_midi,
    load_score,
    notes_to_midi,
    quantize_midi,
    select_melody_instrument,
    skyline,
)
from musegen.theory import Note


def _pm_with_quarters(tempo, pitches):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(0)
    beat = 60.0 / tempo
    for i, p in enumerate(pitches):
        inst.notes.append(pretty_midi.Note(90, p, i * beat, (i + 1) * beat))
    pm.instruments.append(inst)
    return pm


def test_quantization_is_tempo_aware():
    # The notebook used fs=4 frames/second, so a quarter note at 60 bpm became 4 frames but
    # at 180 bpm only ~1.3 frames. On a beat grid, a quarter is always 4 sixteenth steps.
    for tempo in (60, 120, 180):
        score = quantize_midi(_pm_with_quarters(tempo, [60, 62, 64, 65]), steps_per_beat=4)
        assert [(n.start, n.duration) for n in score.notes] == [(0, 4), (4, 4), (8, 4), (12, 4)]
        assert abs(score.tempo - tempo) < 1e-6


def test_skyline_is_monophonic_and_keeps_top_voice():
    chordal = [Note(0, 60, 8), Note(0, 64, 8), Note(0, 67, 8), Note(4, 72, 4), Note(8, 55, 4)]
    line = skyline(chordal)
    starts = [n.start for n in line]
    assert len(starts) == len(set(starts))
    assert line[0].pitch == 67 and line[0].duration == 4  # truncated by the next onset
    assert all(a.end <= b.start for a, b in zip(line, line[1:], strict=False))


def test_melody_instrument_ignores_drums_and_empty_tracks():
    lead = pretty_midi.Instrument(0, name="lead")
    lead.notes = [pretty_midi.Note(80, 72 + i % 3, i * 0.5, i * 0.5 + 0.5) for i in range(20)]
    bass = pretty_midi.Instrument(33, name="bass")
    bass.notes = [pretty_midi.Note(80, 40, i * 0.5, i * 0.5 + 0.5) for i in range(20)]
    drums = pretty_midi.Instrument(0, is_drum=True)
    drums.notes = [pretty_midi.Note(80, 81, i * 0.5, i * 0.5 + 0.1) for i in range(20)]
    empty = pretty_midi.Instrument(0)
    assert select_melody_instrument([bass, drums, empty, lead]).name == "lead"
    assert select_melody_instrument([drums, empty]) is None


def test_write_then_read_round_trip(tmp_path):
    notes = [Note(0, 60, 4, 100), Note(4, 60, 4, 100), Note(8, 67, 8, 100)]
    path = tmp_path / "rt.mid"
    notes_to_midi(notes, steps_per_beat=4, tempo=100).write(str(path))
    score = load_score(path)
    # repeated pitch stays two notes, held note stays one note
    assert [(n.start, n.pitch, n.duration) for n in score.notes] == [
        (n.start, n.pitch, n.duration) for n in notes
    ]


def test_invalid_key_signature_is_repaired(tmp_path):
    # Reproduces the notebook's "Could not decode key with 16 sharps and mode 1" failure.
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("key_signature", key="A", time=0))
    for i in range(8):
        track.append(mido.Message("note_on", note=60 + i, velocity=90, time=0 if i == 0 else 240))
        track.append(mido.Message("note_off", note=60 + i, velocity=0, time=240))
    mid.tracks.append(track)
    path = tmp_path / "bad_key.mid"
    mid.save(str(path))
    raw = path.read_bytes()
    good = bytes([0xFF, 0x59, 0x02, 0x03, 0x00])
    assert good in raw
    path.write_bytes(raw.replace(good, bytes([0xFF, 0x59, 0x02, 0x10, 0x01])))

    pm = load_midi(path)
    assert len(pm.instruments[0].notes) == 8
