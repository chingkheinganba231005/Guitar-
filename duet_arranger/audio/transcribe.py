"""Note transcription of each stem with Spotify's Basic Pitch (ONNX runtime)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# Per-stem settings: (onset threshold, frame threshold, min note ms, fmin, fmax)
STEM_SETTINGS = {
    "vocals": (0.55, 0.35, 90.0, 75.0, 1200.0),
    "bass": (0.45, 0.3, 80.0, 28.0, 420.0),
    "piano": (0.5, 0.3, 70.0, 27.0, 4200.0),
    "guitar": (0.5, 0.3, 70.0, 70.0, 1800.0),
    "other": (0.55, 0.33, 90.0, 50.0, 3000.0),
}


@dataclass
class NoteEvent:
    start: float  # seconds
    end: float
    pitch: int
    amplitude: float
    bends: list = field(default_factory=list)
    tags: set = field(default_factory=set)


_model = None


def _get_model():
    global _model
    if _model is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model

        _model = Model(ICASSP_2022_MODEL_PATH)
    return _model


def stem_is_silent(path: str, threshold_db: float = -45.0) -> bool:
    import soundfile as sf

    data, _ = sf.read(path, always_2d=True, dtype="float32")
    rms = float(np.sqrt(np.mean(data**2)) + 1e-12)
    return 20 * np.log10(rms) < threshold_db


def transcribe(path: str, stem: str) -> list[NoteEvent]:
    from basic_pitch.inference import predict

    onset, frame, min_ms, fmin, fmax = STEM_SETTINGS.get(stem, STEM_SETTINGS["other"])
    _, _, events = predict(
        path,
        _get_model(),
        onset_threshold=onset,
        frame_threshold=frame,
        minimum_note_length=min_ms,
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        melodia_trick=True,
    )
    out = []
    for st, en, pitch, amp, bends in events:
        ev = NoteEvent(float(st), float(en), int(pitch), float(amp), list(bends or []))
        out.append(ev)
    out.sort(key=lambda e: (e.start, e.pitch))
    if stem == "vocals":
        tag_vocal_inflections(out)
    return clean(out, stem)


def tag_vocal_inflections(events: list[NoteEvent]) -> None:
    """Scoops into a note and falls off its end, from Basic Pitch's pitch bends.

    Bends are in 1/3-semitone bins relative to the note pitch, one per frame
    (~11.6 ms)."""
    for e in events:
        b = e.bends
        if len(b) < 8:
            continue
        head = np.mean(b[:3])
        body = np.median(b[len(b) // 4 : 3 * len(b) // 4])
        tail = np.mean(b[-3:])
        if head - body <= -2.5:
            e.tags.add("scoop")
        if tail - body <= -3.5 and e.end - e.start > 0.25:
            e.tags.add("fall")
        if np.std(b[len(b) // 4 :]) > 0.9 and e.end - e.start > 0.5:
            e.tags.add("vibrato")


def clean(events: list[NoteEvent], stem: str) -> list[NoteEvent]:
    if not events:
        return events
    amps = np.array([e.amplitude for e in events])
    floor = max(0.12, float(np.percentile(amps, 12)) * 0.8)
    out = [e for e in events if e.amplitude >= floor and e.end - e.start >= 0.05]
    if stem in ("vocals", "bass"):
        out = monophonic_events(out, prefer="loud" if stem == "vocals" else "low")
    return out


def monophonic_events(events: list[NoteEvent], prefer: str = "loud") -> list[NoteEvent]:
    """One note at a time; overlapping weaker notes (octave errors, reverb) are removed."""
    events = sorted(events, key=lambda e: e.start)
    out: list[NoteEvent] = []
    for e in events:
        if out and e.start < out[-1].end - 0.03:
            prev = out[-1]
            better = e.amplitude > prev.amplitude * 1.15 if prefer == "loud" else e.pitch < prev.pitch
            if better:
                if e.start - prev.start < 0.06:
                    out[-1] = e
                    continue
                prev.end = e.start
            else:
                if e.end <= prev.end + 0.03:
                    continue
                e.start = prev.end
                if e.end - e.start < 0.06:
                    continue
        out.append(e)
    return out
