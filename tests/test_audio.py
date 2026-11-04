"""End-to-end audio test (Demucs + Basic Pitch). Slow: run with DUET_AUDIO_TESTS=1."""

import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("DUET_AUDIO_TESTS"), reason="set DUET_AUDIO_TESTS=1 to run the audio pipeline")


def test_audio_pipeline(tmp_path):
    from duet_arranger.audio.pipeline import analyze_audio
    from duet_arranger.demo import demo_song
    from duet_arranger.synth import render

    wav = tmp_path / "pop.wav"
    render(demo_song("pop"), str(wav))
    song = analyze_audio(str(wav), str(tmp_path / "work"))
    assert abs(song.bars[1].tempo - 96) < 4
    assert song.bars[1].meter == (4, 4)
    assert (song.key_tonic, song.key_mode) == (7, "major")
    assert song.percussive
    names = {c.name() for c in song.chords}
    assert {"G", "D", "Em", "C"} <= {n.split("/")[0] for n in names}
