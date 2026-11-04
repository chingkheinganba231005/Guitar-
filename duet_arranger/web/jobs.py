"""Background jobs for the web app: one heavy job at a time, kept on disk."""

from __future__ import annotations

import json
import logging
import os
import pickle
import queue
import re
import secrets
import shutil
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..progress import STAGES, SYMBOLIC_STAGES, Cancelled, overall

log = logging.getLogger(__name__)

STEM_LABELS = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "piano": "Piano",
    "other": "Other instruments",
}


@dataclass
class Job:
    id: str
    kind: str  # audio | midi | demo | rearrange
    title: str = ""
    artist: str = ""
    input_name: str = ""
    options: dict = field(default_factory=dict)
    state: str = "queued"  # queued | running | done | error | cancelled
    stage: str = ""
    stage_label: str = ""
    stage_progress: float = 0.0
    progress: float = 0.0
    message: str = "Waiting for the arranger"
    error: str = ""
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    parent: str = ""  # job whose analysis a re-arrangement reuses
    stages: list = field(default_factory=list)
    result: dict = field(default_factory=dict)
    files: list = field(default_factory=list)

    def public(self) -> dict:
        d = asdict(self)
        d["elapsed"] = round((self.finished or time.time()) - self.started, 1) if self.started else 0.0
        return d


def safe_name(name: str, fallback: str = "song") -> str:
    base = re.sub(r"[^A-Za-z0-9 _.-]", "", name or "").strip().replace(" ", "-")
    return base[:80] or fallback


class JobManager:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.cancelled: set[str] = set()
        self._load()
        self.worker = threading.Thread(target=self._work, name="duet-arranger-worker", daemon=True)
        self.worker.start()

    # ------------------------------------------------------------------ storage
    def job_dir(self, job_id: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{16}", job_id or ""):
            raise KeyError(job_id)
        return os.path.join(self.root, job_id)

    def _load(self) -> None:
        for name in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, name, "job.json")
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                job = Job(**{k: v for k, v in data.items() if k in Job.__dataclass_fields__})
            except Exception:  # pragma: no cover - a damaged job file is skipped
                continue
            if job.state in ("queued", "running"):
                job.state = "error"
                job.error = "The server stopped before this job finished. Start it again."
            self.jobs[job.id] = job

    def save(self, job: Job) -> None:
        d = self.job_dir(job.id)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "job.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(job), f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(d, "job.json"))

    # ------------------------------------------------------------------ api
    def create(self, kind: str, **kw) -> Job:
        job = Job(id=secrets.token_hex(8), kind=kind, **kw)
        stages = SYMBOLIC_STAGES if kind in ("midi", "demo", "rearrange") else [s for s, _, _ in STAGES]
        job.stages = [{"id": s, "label": l} for s, _, l in STAGES if s in stages]
        with self.lock:
            self.jobs[job.id] = job
        os.makedirs(self.job_dir(job.id), exist_ok=True)
        self.save(job)
        return job

    def submit(self, job: Job) -> Job:
        self.queue.put(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: -j.created)

    def cancel(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job and job.state in ("queued", "running"):
            self.cancelled.add(job_id)
            if job.state == "queued":
                job.state = "cancelled"
                job.message = "Cancelled"
                self.save(job)

    def delete(self, job_id: str) -> None:
        self.cancel(job_id)
        with self.lock:
            self.jobs.pop(job_id, None)
        d = self.job_dir(job_id)
        children = [j for j in self.jobs.values() if j.parent == job_id]
        if children:
            # keep the analysis that re-arrangements still read from
            for name in os.listdir(d) if os.path.isdir(d) else []:
                if name not in ("stems", "song.pkl", "analysis.json", "input"):
                    p = os.path.join(d, name)
                    shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
            return
        shutil.rmtree(d, ignore_errors=True)

    def file_path(self, job_id: str, rel: str) -> Optional[str]:
        """Resolve a result file of a job (stems may live with the parent job)."""
        job = self.jobs.get(job_id)
        if job is None:
            return None
        for owner in (job_id, job.parent):
            if not owner:
                continue
            base = self.job_dir(owner)
            p = os.path.realpath(os.path.join(base, rel))
            if not p.startswith(os.path.realpath(base) + os.sep):
                return None
            if os.path.isfile(p):
                return p
        return None

    # ------------------------------------------------------------------ worker
    def _work(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.state == "cancelled":
                continue
            try:
                self._run(job)
            except Exception:  # pragma: no cover - _run records its own errors
                log.exception("job %s crashed", job_id)

    def _progress(self, job: Job):
        last = [0.0]
        names = [s["id"] for s in job.stages]
        labels = {s: l for s, _, l in STAGES}

        def cb(stage: str, fraction: float = 0.0, message: str = "") -> None:
            if job.id in self.cancelled:
                raise Cancelled()
            job.stage = stage
            job.stage_label = labels.get(stage, stage)
            job.stage_progress = round(fraction, 3)
            job.progress = round(overall(stage, fraction, names), 3)
            job.message = message or job.stage_label
            now = time.time()
            if now - last[0] > 0.7 or fraction >= 1.0:
                last[0] = now
                self.save(job)

        return cb

    def _run(self, job: Job) -> None:
        from ..arrange.duet import ArrangeOptions, arrange
        from ..cli import write_outputs
        from ..model import reserve_ids

        d = self.job_dir(job.id)
        job.state = "running"
        job.started = time.time()
        job.message = "Starting"
        self.save(job)
        progress = self._progress(job)
        try:
            o = job.options
            if job.kind == "demo":
                from ..demo import demo_song

                progress("analyse", 0.5, "Building the demo song")
                song = demo_song(o.get("style", "pop"))
                progress("analyse", 1.0, "Demo song ready")
            elif job.kind == "midi":
                from ..midi_input import read_midi

                progress("analyse", 0.2, "Reading the MIDI tracks")
                song = read_midi(os.path.join(d, "input", job.input_name), job.title or None)
                progress("analyse", 1.0, "Key, chords and sections found")
            elif job.kind == "audio":
                from ..audio.pipeline import analyze_audio

                song = analyze_audio(
                    os.path.join(d, "input", job.input_name),
                    d,
                    title=job.title or None,
                    artist=job.artist,
                    device=o.get("device") or None,
                    progress=progress,
                )
            elif job.kind == "rearrange":
                song = self._parent_song(job, progress)
            else:
                raise ValueError(f"Unknown job kind {job.kind}")
            if job.title:
                song.title = job.title
            if job.artist:
                song.artist = job.artist
            if job.kind != "rearrange":
                save_song(d, song)
            opts = ArrangeOptions(
                tunings=o.get("tuning", "auto") or "auto",
                capo=str(o.get("capo", "auto") or "auto"),
                max_capo=int(o.get("max_capo", 7)),
                max_fret=int(o.get("max_fret", 20)),
                percussion=o.get("percussion", "auto") or "auto",
                harmonics=bool(o.get("harmonics", True)),
                taps=bool(o.get("taps", True)),
                difficulty=float(o.get("difficulty", 1.0)),
            )
            arr = arrange(song, opts, progress)
            base = safe_name(job.title or song.title, "arrangement")
            result = write_outputs(arr, d, base, job.title or None, job.artist or None, progress)
            job.result = summarise(arr, result)
            job.files = self._files(job, base)
            job.state = "done"
            job.progress = 1.0
            job.message = "Your duet is ready"
        except Cancelled:
            job.state = "cancelled"
            job.message = "Cancelled"
        except Exception as err:
            job.state = "error"
            job.error = f"{type(err).__name__}: {err}"
            job.message = "Something went wrong"
            with open(os.path.join(d, "error.log"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            log.exception("job %s failed", job.id)
        finally:
            job.finished = time.time()
            self.cancelled.discard(job.id)
            self.save(job)

    def _parent_song(self, job: Job, progress):
        """The analysis a re-arrangement starts from, redone if it is outdated."""
        from ..audio.pipeline import ANALYSIS_VERSION
        from ..model import reserve_ids

        pdir = self.job_dir(job.parent)
        parent = self.jobs.get(job.parent)
        version = 0
        try:
            with open(os.path.join(pdir, "analysis.version"), encoding="utf-8") as f:
                version = int(f.read().strip() or 0)
        except (OSError, ValueError):
            pass
        if version != ANALYSIS_VERSION and parent is not None and parent.input_name:
            src = os.path.join(pdir, "input", parent.input_name)
            if parent.kind == "audio" and os.path.exists(src):
                from ..audio.pipeline import analyze_audio

                job.stages = [{"id": s, "label": l} for s, _, l in STAGES if s != "separate"]
                self.save(job)
                song = analyze_audio(src, pdir, title=parent.title or None, artist=parent.artist, progress=progress)
                save_song(pdir, song)
                return song
            if parent.kind == "midi" and os.path.exists(src):
                from ..midi_input import read_midi

                progress("analyse", 0.3, "Reading the MIDI file again with the newer analysis")
                song = read_midi(src, parent.title or None)
                save_song(pdir, song)
                progress("analyse", 1.0, "Analysis ready")
                return song
        progress("analyse", 0.3, "Reusing the separated stems and analysis")
        with open(os.path.join(pdir, "song.pkl"), "rb") as f:
            song = pickle.load(f)
        reserve_ids(_max_id(song))
        progress("analyse", 1.0, "Analysis loaded")
        return song

    def _files(self, job: Job, base: str) -> list[dict]:
        out = []
        for name, label, kind in (
            (f"{base}.atelier.json", "Atelier duet", "atelier"),
            (f"{base}.tab.txt", "Text tab", "tab"),
            (f"{base}.preview.mid", "MIDI preview", "midi"),
            (f"{base}.report.json", "Arrangement report", "report"),
            ("analysis.json", "Song analysis", "analysis"),
        ):
            if self.file_path(job.id, name):
                out.append({"name": name, "label": label, "kind": kind})
        owner = job.parent or job.id
        stem_dir = os.path.join(self.job_dir(owner), "stems")
        if os.path.isdir(stem_dir):
            for s, label in STEM_LABELS.items():
                if os.path.exists(os.path.join(stem_dir, f"{s}.wav")):
                    out.append({"name": f"stems/{s}.wav", "label": label, "kind": "stem"})
        src = job if job.kind == "audio" else (self.jobs.get(job.parent) if job.parent else None)
        if src is not None and src.kind == "audio" and self.file_path(src.id, f"input/{src.input_name}"):
            out.insert(0, {"name": f"input/{src.input_name}", "label": "Original recording", "kind": "original"})
        return out


def save_song(d: str, song) -> None:
    from ..audio.pipeline import ANALYSIS_VERSION

    with open(os.path.join(d, "song.pkl"), "wb") as f:
        pickle.dump(song, f)
    with open(os.path.join(d, "analysis.version"), "w", encoding="utf-8") as f:
        f.write(str(ANALYSIS_VERSION))


def _max_id(song) -> int:
    m = 0
    for ns in list(song.parts.values()) + [song.melody, song.bass]:
        for n in ns:
            m = max(m, n.id)
    for h in song.hits:
        m = max(m, h.id)
    return m


def summarise(arr, result: dict) -> dict:
    from ..theory import pc_name, prefers_flats

    song = arr.song
    flats = prefers_flats(song.key_tonic, song.key_mode)
    guitars = []
    for g in arr.guitars:
        st = g.stats
        guitars.append(
            {
                "name": g.name,
                "role": "Melody, harmony under it, fills and harmonics" if g.role == "lead" else ("Bass, the original comping rhythm and percussion" if song.percussive else "Bass and the original comping rhythm"),
                "tuning": g.tuning.name,
                "pitches": g.tuning.label,
                "capo": g.capo,
                "notes": st.get("notes", 0),
                "harmonics": sum(1 for n in g.placed if n.harmonic),
                "taps": sum(1 for n in g.placed if "RH" in n.techs),
                "open_strings": sum(1 for n in g.placed if n.fret == 0 and not n.harmonic),
                "percussive": sum(1 for n in g.placed if set(n.techs) & {"SB", "WT", "BW"}),
                "dropped": st.get("dropped", 0),
                "considered": st.get("considered", []),
            }
        )
    return {
        "title": song.title,
        "artist": song.artist,
        "key": f"{pc_name(song.key_tonic, flats)} {song.key_mode}",
        "tempo": round(song.bars[0].tempo) if song.bars else None,
        "meter": "/".join(map(str, song.bars[0].meter)) if song.bars else "",
        "bars": len(song.bars),
        "percussive": song.percussive,
        "drum_prominence": round(song.drum_prominence, 2),
        "sections": [{"bar": s.start_bar + 1, "label": s.label} for s in song.sections],
        "chords": sorted({c.name(flats) for c in song.chords}),
        "guitars": guitars,
        "issues": [str(i) for i in result["issues"]],
        "incomplete_bars": result["fill"],
    }
