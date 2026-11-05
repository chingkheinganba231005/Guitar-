"""Stem Studio: the analysis (on a synthesised song delivered as a video), the
chord labelling and the web API."""

import importlib.util
import json
import os
import shutil
import subprocess
import time

import numpy as np
import pytest

HAVE_AUDIO = all(importlib.util.find_spec(m) for m in ("torch", "demucs", "basic_pitch", "onnxruntime")) and shutil.which("ffmpeg")


def test_frame_times_follow_the_beats():
    from stem_studio.analyze import frame_times

    beats = [0.5, 1.0, 1.5, 2.0]
    f = frame_times(beats, 3.0)
    assert f[0] == 0.0 and 0.5 in f and 1.5 in f and f[-1] < 3.0
    slow = frame_times([0.0, 1.0, 2.0], 3.0)
    assert 0.5 in slow  # slow pulses are halved


def test_chords_from_chroma_and_silence():
    from duet_arranger.theory import chord_pcs
    from stem_studio.analyze import label_chords

    frames = [i * 0.5 for i in range(16)]
    C = np.zeros((16, 12))
    for k, (r, q) in enumerate([(7, "maj")] * 4 + [(2, "maj")] * 4 + [(4, "min")] * 4 + [(0, "maj")] * 4):
        for p in chord_pcs(r, q):
            C[k, p] = 1.0
    active = np.ones(16, dtype=bool)
    active[12:] = False  # the instrument stops
    segs = label_chords(C, active, frames, 8.0, (7, "major"), False)
    assert [s[2] for s in segs] == ["G", "D", "Em"]
    assert segs[0][0] == 0.0 and segs[-1][1] == 6.0


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from stem_studio import analyze, server

    def fake_analyse(src, workdir, progress=None, device=None, check_cancel=None):
        assert os.path.exists(src)
        for stage in ("decode", "separate", "transcribe", "rhythm", "chords", "encode"):
            progress(stage, 1.0, stage)
        os.makedirs(os.path.join(workdir, "audio"), exist_ok=True)
        for name in ("original", "piano"):
            with open(os.path.join(workdir, "audio", f"{name}.mp3"), "wb") as f:
                f.write(b"ID3fake")
        res = {
            "version": 1, "duration": 4.0, "original": "audio/original.mp3", "key": "G major", "flats": False,
            "beats": [0, 0.5, 1, 1.5], "bars": [0, 2], "beats_per_bar": 4, "meter": "4/4", "tempo": 120, "tempo_label": "♩ = 120",
            "chords": [[0, 2, "G"], [2, 4, "D"]],
            "tracks": [{"id": "piano", "label": "Piano / keys", "show": "notes+chords", "audio": "audio/piano.mp3", "silent": False, "level": 1.0, "notes": [[0, 1, 67, 0.8]], "chords": [[0, 2, "G"]], "hits": []}],
        }
        with open(os.path.join(workdir, "analysis.json"), "w") as f:
            json.dump(res, f)
        return res

    monkeypatch.setattr(analyze, "analyse", fake_analyse)
    monkeypatch.setattr(server, "audio_support", lambda: {"available": True, "missing": []})
    app = server.create_app(str(tmp_path / "jobs"))
    return TestClient(app)


def wait(client, job_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_upload_analyse_and_serve(client):
    r = client.get("/")
    assert r.status_code == 200 and "Stem Studio" in r.text
    info = client.get("/api/info").json()
    assert ".mp4" in info["accept"] and ".mp3" in info["accept"]
    job = client.post("/api/jobs", files={"file": ("My Song.mp4", b"\x00" * 100, "video/mp4")}, data={"title": ""}).json()
    assert job["title"] == "My Song"
    done = wait(client, job["id"])
    assert done["state"] == "done", done["error"]
    assert done["summary"]["key"] == "G major" and done["duration"] == 4.0
    a = client.get(f"/api/jobs/{job['id']}/analysis").json()
    assert a["tracks"][0]["notes"] == [[0, 1, 67, 0.8]]
    au = client.get(f"/api/jobs/{job['id']}/audio/piano.mp3")
    assert au.status_code == 200 and au.headers["content-type"] == "audio/mpeg"
    assert client.get(f"/api/jobs/{job['id']}/audio/..%2Fjob.json").status_code == 404
    dl = client.get(f"/api/jobs/{job['id']}/audio/original.mp3?download=1")
    assert "attachment" in dl.headers["content-disposition"]
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_rejects_unsupported_and_empty_files(client):
    assert client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/api/jobs", files={"file": ("song.mp3", b"", "audio/mpeg")}).status_code == 400


@pytest.mark.skipif(not HAVE_AUDIO or os.environ.get("DUET_SKIP_AUDIO_TESTS"), reason="audio extras or ffmpeg not installed")
def test_a_video_is_separated_and_transcribed(tmp_path):
    from duet_arranger.demo import demo_song
    from duet_arranger.synth import render
    from stem_studio.analyze import analyse

    wav = tmp_path / "pop.wav"
    render(demo_song("pop"), str(wav))
    mp4 = tmp_path / "pop.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5", "-i", str(wav), "-shortest", "-c:v", "libx264", "-c:a", "aac", str(mp4)], check=True)
    res = analyse(str(mp4), str(tmp_path / "job"))
    assert res["meter"] == "4/4" and res["key"] == "G major"
    assert abs(res["tempo"] - 96) <= 3
    tracks = {t["id"]: t for t in res["tracks"]}
    assert set(tracks) == {"vocals", "piano", "guitar", "other", "bass", "drums"}
    assert len(tracks["piano"]["notes"]) > 40 and len(tracks["bass"]["notes"]) > 20
    assert len(tracks["drums"]["hits"]) > 20
    names = {c[2].split("/")[0] for c in tracks["piano"]["chords"]}
    assert {"G", "D", "Em", "C"} <= names
    for t in res["tracks"]:
        assert os.path.getsize(tmp_path / "job" / t["audio"]) > 1000
    assert os.path.exists(tmp_path / "job" / "audio" / "original.mp3")
    assert not os.path.exists(tmp_path / "job" / "mix.wav")
