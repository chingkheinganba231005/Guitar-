"""Stem Studio web app: upload a song or video, then play it with every
instrument separated, on its own volume, and see what each one plays."""

from __future__ import annotations

import importlib.util
import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .analyze import AUDIO_EXT, VIDEO_EXT, TRACKS
from .jobs import JobManager, safe_name

STATIC = Path(__file__).parent / "static"


def audio_support() -> dict:
    import shutil

    mods = {"torch": "torch", "demucs": "demucs", "basic_pitch": "basic-pitch", "onnxruntime": "onnxruntime", "librosa": "librosa"}
    missing = [pkg for mod, pkg in mods.items() if importlib.util.find_spec(mod) is None]
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    return {"available": not missing, "missing": missing}


def create_app(jobs_root: str = "studio-jobs", token: Optional[str] = None, max_upload_mb: int = 1024, device: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Stem Studio", version=__version__, docs_url="/api/docs", redoc_url=None)
    manager = JobManager(jobs_root, device)
    app.state.manager = manager
    token = token or os.environ.get("STUDIO_ACCESS_TOKEN") or None

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if token and path.startswith("/api/") and path != "/api/info" and not path.startswith("/api/docs"):
            given = request.headers.get("x-access-token") or request.query_params.get("token") or ""
            if not secrets.compare_digest(given, token):
                return JSONResponse({"detail": "Access token required"}, status_code=401)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def job_or_404(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "No such song")
        return job

    @app.get("/api/info")
    def info():
        return {
            "version": __version__,
            "audio": audio_support(),
            "auth_required": bool(token),
            "max_upload_mb": max_upload_mb,
            "accept": sorted(AUDIO_EXT | VIDEO_EXT),
            "tracks": [{"id": i, "label": label, "show": show} for i, label, show in TRACKS],
        }

    @app.get("/api/jobs")
    def list_jobs():
        return [j.public() for j in manager.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return job_or_404(job_id).public()

    @app.post("/api/jobs")
    async def create_job(file: UploadFile = File(...), title: str = Form("")):
        name = os.path.basename(file.filename or "song")
        ext = os.path.splitext(name)[1].lower()
        if ext not in AUDIO_EXT | VIDEO_EXT:
            raise HTTPException(400, f"Unsupported file type {ext or '(none)'}. Use a song (mp3, wav, m4a, flac …) or a video (mp4, mov, mkv, webm …).")
        sup = audio_support()
        if not sup["available"]:
            raise HTTPException(400, "Separation needs " + ", ".join(sup["missing"]) + ". See the README.")
        stored = safe_name(os.path.splitext(name)[0]) + ext
        job = manager.create((title or os.path.splitext(name)[0]).strip()[:120], stored)
        dest = os.path.join(manager.job_dir(job.id), "input")
        os.makedirs(dest, exist_ok=True)
        limit = max_upload_mb * 1024 * 1024
        size = 0
        with open(os.path.join(dest, stored), "wb") as out:
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

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        job_or_404(job_id)
        manager.cancel(job_id)
        return manager.get(job_id).public()

    @app.delete("/api/jobs/{job_id}")
    def delete(job_id: str):
        job = job_or_404(job_id)
        if job.state == "running":
            raise HTTPException(409, "Cancel the analysis before deleting the song.")
        manager.delete(job_id)
        return {"deleted": job_id}

    @app.get("/api/jobs/{job_id}/analysis")
    def analysis(job_id: str):
        job_or_404(job_id)
        p = manager.file_path(job_id, "analysis.json")
        if p is None:
            raise HTTPException(404, "Not analysed yet")
        return FileResponse(p, media_type="application/json")

    @app.get("/api/jobs/{job_id}/audio/{name}")
    def audio(job_id: str, name: str, download: int = 0):
        job = job_or_404(job_id)
        p = manager.file_path(job_id, os.path.join("audio", os.path.basename(name)))
        if p is None:
            raise HTTPException(404, "No such track")
        media = "audio/mpeg" if p.endswith(".mp3") else "audio/wav"
        if download:
            return FileResponse(p, media_type=media, filename=f"{safe_name(job.title)}-{os.path.basename(p)}")
        return FileResponse(p, media_type=media)

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    return app


def serve(host: str = "127.0.0.1", port: int = 8001, jobs: str = "studio-jobs", token: Optional[str] = None, open_browser: bool = False, max_upload_mb: int = 1024, device: Optional[str] = None) -> None:
    import threading
    import webbrowser

    import uvicorn

    app = create_app(jobs, token, max_upload_mb, device)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    print(f"Stem Studio is running at {url}  (Ctrl+C to stop)")
    print(f"Songs are kept in {app.state.manager.root}")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
