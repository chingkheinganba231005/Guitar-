import mido

from duet_arranger.arrange.duet import ArrangeOptions, arrange
from duet_arranger.demo import demo_song
from duet_arranger.export.atelier import to_atelier
from duet_arranger.midi_input import read_midi
from duet_arranger.validate import audit_project


def write_song_midi(song, path):
    mid = mido.MidiFile(ticks_per_beat=480)
    k = 480 // 24
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(song.bars[0].tempo)))
    conductor.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    mid.tracks.append(conductor)
    for name, notes, ch, prog in (("Lead Vocal", song.melody, 0, 54), ("Bass", song.bass, 1, 33), ("Piano", song.parts["piano"], 2, 0)):
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name))
        tr.append(mido.Message("program_change", channel=ch, program=prog))
        ev = []
        for n in notes:
            ev.append((n.start * k, 1, mido.Message("note_on", channel=ch, note=n.pitch, velocity=90)))
            ev.append((n.end * k, 0, mido.Message("note_off", channel=ch, note=n.pitch, velocity=0)))
        ev.sort(key=lambda e: (e[0], e[1]))
        now = 0
        for t, _, m in ev:
            m.time = int(t - now)
            now = t
            tr.append(m)
        mid.tracks.append(tr)
    tr = mido.MidiTrack()
    ev = []
    for h in song.hits:
        note = {"kick": 36, "snare": 38, "hat": 42}[h.kind]
        ev.append((h.start * k, 1, mido.Message("note_on", channel=9, note=note, velocity=100)))
        ev.append((h.start * k + 60, 0, mido.Message("note_off", channel=9, note=note, velocity=0)))
    ev.sort(key=lambda e: (e[0], e[1]))
    now = 0
    for t, _, m in ev:
        m.time = int(t - now)
        now = t
        tr.append(m)
    mid.tracks.append(tr)
    mid.save(path)


def test_midi_roundtrip(tmp_path):
    src = demo_song("pop")
    path = tmp_path / "song.mid"
    write_song_midi(src, str(path))
    song = read_midi(str(path))
    assert len(song.bars) == len(src.bars)
    assert song.key_tonic == 7 and song.key_mode == "major"
    assert len(song.melody) == len(src.melody)
    assert song.percussive
    assert {c.name() for c in song.chords} >= {"G", "D", "Em", "C"}
    arr = arrange(song, ArrangeOptions())
    assert audit_project(to_atelier(arr)) == []
