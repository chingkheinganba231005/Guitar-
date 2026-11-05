"""End-to-end audio tests: synthesised recordings through Demucs, Basic Pitch,
beat tracking and the arranger. They run whenever the audio extras are
installed (about a minute each on a CPU); set DUET_SKIP_AUDIO_TESTS=1 to skip."""

import importlib.util
import os
from collections import Counter

import pytest

HAVE_AUDIO = all(importlib.util.find_spec(m) for m in ("torch", "demucs", "basic_pitch", "onnxruntime"))
pytestmark = pytest.mark.skipif(not HAVE_AUDIO or os.environ.get("DUET_SKIP_AUDIO_TESTS"), reason="audio extras not installed")


def _analyse(tmp_path, style):
    from duet_arranger.audio.pipeline import analyze_audio
    from duet_arranger.demo import demo_song
    from duet_arranger.synth import render

    wav = tmp_path / f"{style}.wav"
    render(demo_song(style), str(wav))
    return analyze_audio(str(wav), str(tmp_path / "work"))


def test_pop_recording(tmp_path):
    song = _analyse(tmp_path, "pop")
    assert abs(song.bars[1].tempo - 96) < 4
    assert song.bars[1].meter == (4, 4)
    assert (song.key_tonic, song.key_mode) == (7, "major")
    assert song.percussive
    names = {c.name() for c in song.chords}
    assert {"G", "D", "Em", "C"} <= {n.split("/")[0] for n in names}


def test_waltz_recording_arranges_cleanly(tmp_path):
    from duet_arranger.arrange.duet import ArrangeOptions, arrange
    from duet_arranger.export.atelier import to_atelier
    from duet_arranger.validate import audit_project, bar_fill_errors

    song = _analyse(tmp_path, "waltz")
    assert song.bars[0].meter == (3, 4)  # no pickup bar: the song starts on beat 1
    assert all(b.meter == (3, 4) for b in song.bars)
    assert abs(song.bars[0].tempo - 136) < 5
    assert len({b.tempo for b in song.bars}) == 1
    assert (song.key_tonic, song.key_mode) == (10, "major")
    assert not song.percussive
    arr = arrange(song, ArrangeOptions())
    p = to_atelier(arr)
    assert audit_project(p) == [] and bar_fill_errors(p) == []
    for t in p["tracks"]:
        evs = [e for b in t["bars"] for e in b["events"]]
        assert not any(e["den"] >= 32 for e in evs), t["name"]
        attacks = [n for e in evs for n in e["notes"] if not n.get("tie")]
        assert sum(n["sustain"] for n in attacks) / len(attacks) > 0.9
        assert sum(1 for e in evs if not e["notes"]) <= 2
    lead = arr.guitars[0].placed
    intro_end = song.bars[4].start
    onsets = sorted({n.start for n in lead if n.start < intro_end})
    assert onsets and all(t % 24 == 0 for t in onsets)  # on the beats
    per_bar = Counter(song.bar_at(t) for t in onsets)
    assert max(per_bar.values()) <= 3
    harm = Counter(n.fret for g in arr.guitars for n in g.placed if n.harmonic == "natural")
    assert harm.get(5, 0) == 0
