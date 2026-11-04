"""Progress reporting shared by the CLI and the web app."""

from __future__ import annotations

from typing import Callable, Optional

# (stage, share of the whole job, label) for an audio job
STAGES = [
    ("separate", 0.46, "Separating vocals, drums, bass, guitar, piano and other"),
    ("transcribe", 0.2, "Transcribing the notes of every stem"),
    ("rhythm", 0.04, "Finding the beat, the meter and the bars"),
    ("analyse", 0.04, "Reading key, chords, sections and drums"),
    ("guitar1", 0.11, "Arranging Guitar I · melody"),
    ("guitar2", 0.12, "Arranging Guitar II · bass, harmony and groove"),
    ("export", 0.03, "Writing the tabs and checking every hand position"),
]
SYMBOLIC_STAGES = ("analyse", "guitar1", "guitar2", "export")

ProgressFn = Callable[[str, float, str], None]


class Cancelled(Exception):
    """Raised from a progress callback to stop a running job."""


def null_progress(stage: str, fraction: float = 0.0, message: str = "") -> None:
    return None


def report(progress: Optional[ProgressFn], stage: str, fraction: float = 0.0, message: str = "") -> None:
    if progress is not None:
        progress(stage, max(0.0, min(1.0, fraction)), message)


def overall(stage: str, fraction: float, stages: tuple | list | None = None) -> float:
    """Whole-job completion for ``stage`` at ``fraction`` (stages not run are skipped)."""
    names = [s for s, _, _ in STAGES if stages is None or s in stages]
    weights = {s: w for s, w, _ in STAGES if s in names}
    total = sum(weights.values()) or 1.0
    done = 0.0
    for s in names:
        if s == stage:
            return (done + weights[s] * fraction) / total
        done += weights[s]
    return 1.0
