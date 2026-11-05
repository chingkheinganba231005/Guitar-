import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from duet_arranger.demo import demo_song  # noqa: E402
from duet_arranger.web.jobs import JobManager  # noqa: E402
from duet_arranger.web.server import create_app  # noqa: E402
from tests.test_midi import write_song_midi  # noqa: E402


def wait(client, job_id, timeout=240, headers=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}", headers=headers or {}).json()
        if job["state"] not in ("queued", "running"):
            return job
        time.sleep(0.3)
    raise AssertionError("job did not finish")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("jobs")
    app = create_app(str(root))
    with TestClient(app) as c:
        c.root = root
        yield c


def test_info_and_pages(client):
    info = client.get("/api/info").json()
    assert info["tunings"] and "atelier_url" in info
    assert client.get("/").status_code == 200
    assert "Duet Arranger" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_demo_job_then_rearrange(client):
    job = client.post("/api/demo", data={"style": "pop"}).json()
    assert job["kind"] == "demo" and [s["id"] for s in job["stages"]] == ["analyse", "guitar1", "guitar2", "export"]
    done = wait(client, job["id"])
    assert done["state"] == "done", done["error"]
    r = done["result"]
    assert r["percussive"] and r["key"] == "G major" and len(r["guitars"]) == 2
    assert r["issues"] == []
    kinds = {f["kind"] for f in done["files"]}
    assert {"atelier", "tab", "midi", "report"} <= kinds
    atelier = next(f for f in done["files"] if f["kind"] == "atelier")
    project = client.get(f"/api/jobs/{job['id']}/files/{atelier['name']}").json()
    assert project["version"] == 2 and len(project["tracks"]) == 2
    tab = next(f for f in done["files"] if f["kind"] == "tab")
    res = client.get(f"/api/jobs/{job['id']}/files/{tab['name']}?download=1")
    assert res.status_code == 200 and "attachment" in res.headers["content-disposition"]
    # arrange again with forced tunings, reusing the saved analysis
    again = client.post(f"/api/jobs/{job['id']}/rearrange", data={"tuning1": "DADGAD", "tuning2": "Open G", "capo": "none", "percussion": "off", "octaves": "false"}).json()
    assert again["parent"] == job["id"]
    assert client.app.state.manager.get(again["id"]).options["octaves"] is False
    done2 = wait(client, again["id"])
    assert done2["state"] == "done", done2["error"]
    assert [g["tuning"] for g in done2["result"]["guitars"]] == ["DADGAD", "Open G"]
    assert not done2["result"]["percussive"]
    assert all(g["percussive"] == 0 for g in done2["result"]["guitars"])
    ids = [j["id"] for j in client.get("/api/jobs").json()]
    assert job["id"] in ids and again["id"] in ids


def test_midi_upload(client, tmp_path):
    path = tmp_path / "song.mid"
    write_song_midi(demo_song("ballad", bars_per_section=2), str(path))
    with open(path, "rb") as f:
        job = client.post("/api/jobs", data={"title": "My ballad", "artist": "Me"}, files={"file": ("song.mid", f, "audio/midi")}).json()
    assert job["kind"] == "midi" and job["title"] == "My ballad"
    done = wait(client, job["id"])
    assert done["state"] == "done", done["error"]
    assert done["result"]["issues"] == []
    assert not done["result"]["percussive"]


def test_rejects_bad_input(client):
    res = client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    res = client.post("/api/jobs", files={"file": ("song.mid", b"", "audio/midi")})
    assert res.status_code == 400
    res = client.post("/api/demo", data={"style": "pop", "tuning1": "Q Q Q Q Q Q"})
    assert res.status_code == 400
    res = client.post("/api/demo", data={"style": "pop", "capo": "99"})
    assert res.status_code == 400
    assert client.get("/api/jobs/0123456789abcdef").status_code == 404


def test_files_cannot_escape_the_job(client):
    job = client.get("/api/jobs").json()[0]
    m = client.app.state.manager
    assert m.file_path(job["id"], "../../etc/passwd") is None
    assert m.file_path(job["id"], "job.json") is not None
    assert client.get(f"/api/jobs/{job['id']}/files/..%2F..%2Fjob.json").status_code == 404


def test_jobs_survive_a_restart(client):
    m = JobManager(str(client.root))
    assert {j.id for j in m.list()} == {j["id"] for j in client.get("/api/jobs").json()}


def test_access_token(tmp_path):
    app = create_app(str(tmp_path), token="s3cret")
    with TestClient(app) as c:
        assert c.get("/api/info").json()["auth_required"] is True
        assert c.get("/api/jobs").status_code == 401
        assert c.get("/api/jobs", headers={"X-Access-Token": "s3cret"}).status_code == 200
        assert c.get("/api/jobs?token=s3cret").status_code == 200


def test_cancel_and_delete(tmp_path):
    app = create_app(str(tmp_path))
    with TestClient(app) as c:
        first = c.post("/api/demo", data={"style": "ballad"}).json()
        second = c.post("/api/demo", data={"style": "pop"}).json()
        c.post(f"/api/jobs/{second['id']}/cancel")
        assert c.get(f"/api/jobs/{second['id']}").json()["state"] == "cancelled"
        if c.get(f"/api/jobs/{first['id']}").json()["state"] == "running":
            assert c.delete(f"/api/jobs/{first['id']}").status_code == 409
        c.post(f"/api/jobs/{first['id']}/cancel")
        assert wait(c, first["id"])["state"] == "cancelled"
        assert c.delete(f"/api/jobs/{first['id']}").status_code == 200
        assert c.get(f"/api/jobs/{first['id']}").status_code == 404


def test_rearrange_redoes_an_outdated_analysis(client, tmp_path):
    import os

    path = tmp_path / "song.mid"
    write_song_midi(demo_song("pop", bars_per_section=2), str(path))
    with open(path, "rb") as f:
        job = client.post("/api/jobs", files={"file": ("song.mid", f, "audio/midi")}).json()
    assert wait(client, job["id"])["state"] == "done"
    d = client.app.state.manager.job_dir(job["id"])
    os.remove(os.path.join(d, "analysis.version"))  # as if made by an older version
    again = client.post(f"/api/jobs/{job['id']}/rearrange", data={"capo": "none"}).json()
    done = wait(client, again["id"])
    assert done["state"] == "done", done["error"]
    from duet_arranger.audio.pipeline import ANALYSIS_VERSION

    assert open(os.path.join(d, "analysis.version")).read().strip() == str(ANALYSIS_VERSION)
