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


def test_six_eight_ballad_is_written_in_six_eight(tmp_path):
    """Triple-time ballads are tracked at the eighth note; they must come out
    in 6/8 at the real tempo, not as half-length 3/4 bars."""
    from duet_arranger.arrange.duet import ArrangeOptions, arrange
    from duet_arranger.export.atelier import to_atelier
    from duet_arranger.validate import audit_project, bar_fill_errors

    song = _analyse(tmp_path, "sixeight")
    assert all(b.meter == (6, 8) for b in song.bars), [b.meter for b in song.bars[:4]]
    assert abs(song.bars[0].tempo - 75) < 5  # quarter = 75: dotted quarter = 50
    assert (song.key_tonic, song.key_mode) == (10, "major")
    firsts = [song.chord_at(b.start).name(True) for b in song.bars[:8] if song.chord_at(b.start)]
    assert firsts[:3] == ["Bb", "Bb", "Gm"], firsts
    arr = arrange(song, ArrangeOptions())
    p = to_atelier(arr)
    assert p["meter"] == [6, 8]
    assert audit_project(p) == [] and bar_fill_errors(p) == []
    # Guitar I strikes the intro chords on the dotted quarters
    lead = arr.guitars[0].placed
    intro = sorted({n.start - song.bars[song.bar_at(n.start)].start for n in lead if n.start < song.bars[8].start})
    assert set(intro) <= {0, 36}, intro
    # Guitar II's verse groove sits on eighth notes of the 6/8 bar
    acc = arr.guitars[1].placed
    verse = [n for n in acc if song.bars[8].start <= n.start < song.bars[16].start]
    assert verse and all((n.start - song.bars[song.bar_at(n.start)].start) % 12 == 0 for n in verse)


def test_arrange_again_reanalyses_an_old_audio_job(tmp_path):
    """The path a user takes after updating: 'Arrange again' on a song made by
    an older version must redo the analysis from the saved stems."""
    import time

    from fastapi.testclient import TestClient

    from duet_arranger.audio.pipeline import ANALYSIS_VERSION
    from duet_arranger.demo import demo_song
    from duet_arranger.synth import render
    from duet_arranger.web.server import create_app

    wav = tmp_path / "song.wav"
    render(demo_song("sixeight"), str(wav))
    app = create_app(str(tmp_path / "jobs"))
    with TestClient(app) as c:
        def wait(jid):
            for _ in range(600):
                j = c.get(f"/api/jobs/{jid}").json()
                if j["state"] not in ("queued", "running"):
                    return j
                time.sleep(0.5)
            raise AssertionError("timeout")

        with open(wav, "rb") as f:
            job = c.post("/api/jobs", files={"file": ("song.wav", f, "audio/wav")}).json()
        assert wait(job["id"])["state"] == "done"
        d = app.state.manager.job_dir(job["id"])
        with open(os.path.join(d, "analysis.version"), "w") as f:
            f.write("1")  # as if analysed by an old version
        again = c.post(f"/api/jobs/{job['id']}/rearrange", data={}).json()
        done = wait(again["id"])
        assert done["state"] == "done", done["error"]
        assert open(os.path.join(d, "analysis.version")).read().strip() == str(ANALYSIS_VERSION)
        atelier = next(x for x in done["files"] if x["kind"] == "atelier")
        p = c.get(f"/api/jobs/{again['id']}/files/{atelier['name']}").json()
        assert p["meter"] == [6, 8]
        assert all(b["tempo"] is None for b in p["tracks"][0]["bars"])  # one steady tempo
        assert p["arranger"]["analysis_version"] == ANALYSIS_VERSION
