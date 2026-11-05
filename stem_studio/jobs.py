"""Background jobs: one heavy analysis at a time, every song kept on disk."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import secrets
import shutil
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# stage -> (share of the total time, label)
STAGES = [
    ("decode", 0.02, "Read the file"),
    ("separate", 0.6, "Separate the instruments"),
    ("transcribe", 0.24, "Listen to every note"),
    ("rhythm", 0.05, "Beat and bars"),
    ("chords", 0.05, "Chords"),
    ("encode", 0.04, "Prepare the tracks"),
]


class Cancelled(Exception):
    pass


def overall(stage: str, frac: float) -> float:
    done = 0.0
    for name, share, _ in STAGES:
        if name == stage:
            return done + share * frac
        done += share
    return done


def safe_name(name: str, default: str = "song") -> str:
    s = re.sub(r"[^A-Za-z0-9 ._-]", "", name).strip().replace(" ", "-")
    return s[:80] or default


@dataclass
class Job:
    id: str
    title: str
    input_name: str
    created: float = field(default_factory=time.time)
    state: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = ""
    progress: float = 0.0
    message: str = "Waiting for its turn"
    error: str = ""
    started: float = 0.0
    finished: float = 0.0
    duration: float = 0.0
    summary: dict = field(default_factory=dict)

    def public(self) -> dict:
        d = asdict(self)
        d["stages"] = [{"id": s, "label": label} for s, _, label in STAGES]
        return d


class JobManager:
    def __init__(self, root: str, device: str | None = None):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.device = device
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.cancel_flags: set[str] = set()
        self._load()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    # ---------------------------------------------------------------- storage
    def job_dir(self, job_id: str) -> str:
        return os.path.join(self.root, job_id)

    def _save(self, job: Job) -> None:
        d = self.job_dir(job.id)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "job.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(job), f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(d, "job.json"))

    def _load(self) -> None:
        for name in os.listdir(self.root):
            p = os.path.join(self.root, name, "job.json")
            if not os.path.isfile(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                job = Job(**{k: v for k, v in data.items() if k in Job.__dataclass_fields__})
            except Exception:  # pragma: no cover - a damaged job file
                continue
            if job.state in ("queued", "running"):
                if job.state == "running":
                    job.state, job.error, job.message = "failed", "The app was restarted while this song was being analysed. Analyse it again.", "Interrupted"
                    self._save(job)
                else:
                    self.queue.put(job.id)
            self.jobs[job.id] = job

    # ---------------------------------------------------------------- api
    def list(self) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: -j.created)

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def create(self, title: str, input_name: str) -> Job:
        job = Job(id=time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3), title=title, input_name=input_name)
        with self.lock:
            self.jobs[job.id] = job
        self._save(job)
        return job

    def submit(self, job: Job) -> None:
        self.queue.put(job.id)

    def cancel(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        if job.state == "queued":
            job.state, job.message = "cancelled", "Cancelled"
            self._save(job)
        elif job.state == "running":
            self.cancel_flags.add(job_id)

    def delete(self, job_id: str) -> None:
        with self.lock:
            self.jobs.pop(job_id, None)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def file_path(self, job_id: str, rel: str) -> Optional[str]:
        base = os.path.realpath(self.job_dir(job_id))
        p = os.path.realpath(os.path.join(base, rel))
        if not p.startswith(base + os.sep) or not os.path.isfile(p):
            return None
        return p

    # ---------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.state != "queued":
                continue
            try:
                self._process(job)
            except Cancelled:
                job.state, job.message = "cancelled", "Cancelled"
            except Exception as err:  # noqa: BLE001 - shown to the user
                log.error("Job %s failed:\n%s", job.id, traceback.format_exc())
                job.state, job.error, job.message = "failed", f"{type(err).__name__}: {err}", "Failed"
            finally:
                self.cancel_flags.discard(job.id)
                job.finished = time.time()
                self._save(job)

    def _process(self, job: Job) -> None:
        from .analyze import analyse

        job.state, job.started, job.message, job.progress = "running", time.time(), "Starting", 0.0
        self._save(job)
        last = [0.0]

        def check():
            if job.id in self.cancel_flags:
                raise Cancelled()

        def progress(stage: str, frac: float, msg: str) -> None:
            check()
            job.stage, job.message = stage, msg
            job.progress = round(max(job.progress, overall(stage, frac)), 4)
            if time.time() - last[0] > 1.0:
                last[0] = time.time()
                self._save(job)

        d = self.job_dir(job.id)
        src = os.path.join(d, "input", job.input_name)
        res = analyse(src, d, progress=progress, device=self.device, check_cancel=check)
        job.duration = res["duration"]
        job.summary = {
            "key": res["key"],
            "meter": res["meter"],
            "tempo": res["tempo_label"],
            "bars": len(res["bars"]),
            "tracks": [t["id"] for t in res["tracks"] if not t["silent"]],
        }
        job.state, job.progress, job.message = "done", 1.0, "Ready"
