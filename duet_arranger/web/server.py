"""Duet Arranger web app: upload a song, follow the progress, open the duet in Atelier."""

from __future__ import annotations

import importlib.util
import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..cli import AUDIO_EXT, MIDI_EXT, parse_tuning
from ..guitar import NAMED_TUNINGS, Tuning
from .jobs import JobManager, safe_name

STATIC = Path(__file__).parent / "static"
HOSTED_ATELIER = "https://chingkheinganba231005.github.io/atelier-fingerstyle/"


def audio_support() -> dict:
    mods = {"torch": "torch", "demucs": "demucs", "basic_pitch": "basic-pitch", "onnxruntime": "onnxruntime"}
    missing = [pkg for mod, pkg in mods.items() if importlib.util.find_spec(mod) is None]
    return {"available": not missing, "missing": missing}


def find_atelier(explicit: Optional[str] = None) -> Optional[str]:
    cands = [explicit, os.environ.get("ATELIER_DIR")]
    here = Path(__file__).resolve().parents[2]
    cands += [str(here / "atelier"), str(here.parent / "atelier-fingerstyle"), str(Path.cwd() / "atelier-fingerstyle")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "index.html")) and os.path.isfile(os.path.join(c, "app.js")):
            return os.path.abspath(c)
    return None


def _options(form: dict) -> dict:
    """Validate arrangement options coming from the browser."""
    out: dict = {}
    t1 = (form.get("tuning1") or "auto").strip()
    t2 = (form.get("tuning2") or "auto").strip()
    for t in (t1, t2):
        if t.lower() not in ("auto", "standard"):
            try:
                parse_tuning(t)
            except ValueError as err:
                raise HTTPException(400, str(err))
    out["tuning"] = "auto" if t1.lower() == t2.lower() == "auto" else f"{t1},{t2}"
    capo = (form.get("capo") or "auto").strip().lower()
    if capo not in ("auto", "none") and not (capo.isdigit() and 0 <= int(capo) <= 12):
        raise HTTPException(400, "Capo must be auto, none or 0–12")
    out["capo"] = capo
    perc = (form.get("percussion") or "auto").strip().lower()
    if perc not in ("auto", "on", "off"):
        raise HTTPException(400, "Percussion must be auto, on or off")
    out["percussion"] = perc
    out["harmonics"] = str(form.get("harmonics", "true")).lower() in ("1", "true", "on", "yes")
    out["taps"] = str(form.get("taps", "true")).lower() in ("1", "true", "on", "yes")
    try:
        out["max_fret"] = max(12, min(24, int(form.get("max_fret") or 20)))
        out["difficulty"] = max(0.5, min(1.0, float(form.get("difficulty") or 1.0)))
    except ValueError:
        raise HTTPException(400, "Max fret and texture must be numbers")
    if form.get("device"):
        out["device"] = str(form["device"])[:10]
    return out


def create_app(jobs_root: str = "duet-jobs", atelier_dir: Optional[str] = None, atelier_url: Optional[str] = None, token: Optional[str] = None, max_upload_mb: int = 300) -> FastAPI:
    app = FastAPI(title="Duet Arranger", version=__version__, docs_url="/api/docs", redoc_url=None)
    manager = JobManager(jobs_root)
    app.state.manager = manager
    local_atelier = find_atelier(atelier_dir)
    atelier_href = "/atelier/" if local_atelier else (atelier_url or HOSTED_ATELIER)
    token = token or os.environ.get("DUET_ACCESS_TOKEN") or None

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if token and path.startswith("/api/") and path not in ("/api/info",) and not path.startswith("/api/docs") and path != "/openapi.json":
            given = request.headers.get("x-access-token") or request.query_params.get("token") or ""
            if not secrets.compare_digest(given, token):
                return JSONResponse({"detail": "Access token required"}, status_code=401)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if path == "/" or path.startswith("/static/") or path.startswith("/atelier/"):
            # always revalidate app files, so an update is picked up on reload
            response.headers["Cache-Control"] = "no-cache"
        return response

    def job_or_404(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "No such arrangement")
        return job

    @app.get("/api/info")
    def info():
        return {
            "version": __version__,
            "audio": audio_support(),
            "atelier_url": atelier_href,
            "atelier_local": bool(local_atelier),
            "auth_required": bool(token),
            "max_upload_mb": max_upload_mb,
            "accept": sorted(AUDIO_EXT | MIDI_EXT),
            "tunings": [{"name": k, "pitches": Tuning(k, v).label} for k, v in NAMED_TUNINGS.items()],
        }

    @app.get("/api/jobs")
    def list_jobs():
        return [j.public() for j in manager.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return job_or_404(job_id).public()

    @app.post("/api/jobs")
    async def create_job(request: Request, file: UploadFile = File(...), title: str = Form(""), artist: str = Form("")):
        form = dict(await request.form())
        opts = _options(form)
        name = os.path.basename(file.filename or "song")
        ext = os.path.splitext(name)[1].lower()
        if ext in MIDI_EXT:
            kind = "midi"
        elif ext in AUDIO_EXT:
            kind = "audio"
            sup = audio_support()
            if not sup["available"]:
                raise HTTPException(400, "Audio input needs " + ", ".join(sup["missing"]) + ". See the README (pip install -r requirements-audio.txt). MIDI files work already.")
        else:
            raise HTTPException(400, f"Unsupported file type {ext or '(none)'}. Use audio (mp3, wav, flac, m4a, ogg …) or MIDI.")
        stored = safe_name(os.path.splitext(name)[0]) + ext
        song_title = (title or os.path.splitext(name)[0]).strip()[:120]
        job = manager.create(kind, title=song_title, artist=artist.strip()[:120], input_name=stored, options=opts)
        dest_dir = os.path.join(manager.job_dir(job.id), "input")
        os.makedirs(dest_dir, exist_ok=True)
        limit = max_upload_mb * 1024 * 1024
        size = 0
        with open(os.path.join(dest_dir, stored), "wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    out.close()
                    manager.delete(job.id)
                    raise HTTPException(413, f"Files up to {max_upload_mb} MB, please.")
                out.write(chunk)
        if size == 0:
            manager.delete(job.id)
            raise HTTPException(400, "The file is empty.")
        manager.submit(job)
        return job.public()

    @app.post("/api/demo")
    async def demo(request: Request):
        form = dict(await request.form())
        style = form.get("style", "pop")
        if style not in ("pop", "ballad", "waltz"):
            raise HTTPException(400, "Demo style is pop, ballad or waltz")
        opts = _options(form)
        opts["style"] = style
        job = manager.create("demo", title=f"Demo {style}", artist="Duet Arranger", options=opts)
        manager.submit(job)
        return job.public()

    @app.post("/api/jobs/{job_id}/rearrange")
    async def rearrange(job_id: str, request: Request):
        src = job_or_404(job_id)
        root = src.parent or src.id
        if not os.path.exists(os.path.join(manager.job_dir(root), "song.pkl")):
            raise HTTPException(409, "This arrangement has no saved analysis to reuse yet.")
        form = dict(await request.form())
        opts = _options(form)
        job = manager.create("rearrange", title=form.get("title") or src.title, artist=form.get("artist") or src.artist, input_name=src.input_name, options=opts, parent=root)
        manager.submit(job)
        return job.public()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        job_or_404(job_id)
        manager.cancel(job_id)
        return manager.get(job_id).public()

    @app.delete("/api/jobs/{job_id}")
    def delete(job_id: str):
        job = job_or_404(job_id)
        if job.state == "running":
            raise HTTPException(409, "Cancel the arrangement before deleting it.")
        manager.delete(job_id)
        return {"deleted": job_id}

    @app.get("/api/jobs/{job_id}/files/{rel:path}")
    def job_file(job_id: str, rel: str, download: int = 0):
        job_or_404(job_id)
        p = manager.file_path(job_id, rel)
        if p is None:
            raise HTTPException(404, "No such file")
        media = None
        if p.endswith(".json"):
            media = "application/json"
        elif p.endswith(".txt"):
            media = "text/plain; charset=utf-8"
        elif p.endswith(".mid"):
            media = "audio/midi"
        elif p.endswith(".wav"):
            media = "audio/wav"
        name = os.path.basename(p)
        if download:
            return FileResponse(p, media_type=media, filename=name)
        return FileResponse(p, media_type=media)

    if local_atelier:
        app.mount("/atelier", StaticFiles(directory=local_atelier, html=True), name="atelier")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, jobs: str = "duet-jobs", atelier: Optional[str] = None, token: Optional[str] = None, open_browser: bool = False, max_upload_mb: int = 300) -> None:
    import threading
    import webbrowser

    import uvicorn

    atelier_dir = atelier if atelier and not atelier.startswith("http") else None
    atelier_url = atelier if atelier and atelier.startswith("http") else None
    app = create_app(jobs, atelier_dir, atelier_url, token, max_upload_mb)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    print(f"Duet Arranger is running at {url}  (Ctrl+C to stop)")
    a = app.state.manager
    print(f"Arrangements are kept in {a.root}")
    if find_atelier(atelier_dir):
        print(f"Atelier served from {find_atelier(atelier_dir)}")
    else:
        print(f"Atelier: {atelier_url or HOSTED_ATELIER}")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
