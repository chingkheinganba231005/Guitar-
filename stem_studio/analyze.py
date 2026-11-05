"""From a song or video to a "studio" folder the player opens:

* ``original.mp3`` — the recording (stereo), and ``stems/<name>.mp3`` — each
  separated instrument (mono, to keep the browser's memory low);
* ``analysis.json`` — beats and bars, key, tempo, and per instrument the notes
  it plays (in seconds), the chords it plays and, for drums, the hits.

Separation is Demucs htdemucs_6s (vocals, drums, bass, guitar, piano, other —
strings and synths land in "other"); notes are Spotify's Basic Pitch; beats,
meter and chord recognition come from the Duet Arranger's analysis.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

ANALYSIS_VERSION = 1
SR = 22050

TRACKS = [
    # id, label, what the player shows
    ("vocals", "Vocals", "notes"),
    ("piano", "Piano / keys", "notes+chords"),
    ("guitar", "Guitar", "chords"),
    ("other", "Strings / synth", "notes+chords"),
    ("bass", "Bass", "notes"),
    ("drums", "Drums", "hits"),
]
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif", ".opus", ".wma"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp", ".mpg", ".mpeg", ".wmv"}

Progress = Optional[Callable[[str, float, str], None]]


def _report(progress: Progress, stage: str, frac: float, msg: str) -> None:
    if progress is not None:
        progress(stage, float(min(1.0, max(0.0, frac))), msg)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def decode(src: str, out_wav: str) -> float:
    """Any audio or video file to a 44.1 kHz stereo WAV; returns its duration."""
    ffmpeg("-i", src, "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", out_wav)
    import soundfile as sf

    info = sf.info(out_wav)
    if info.frames == 0:
        raise ValueError("The file has no sound track.")
    return info.frames / info.samplerate


def encode(wav: str, out: str, mono: bool, bitrate: str) -> str:
    """WAV to MP3 for the browser (falls back to the WAV itself)."""
    try:
        ffmpeg("-i", wav, *(["-ac", "1"] if mono else []), "-c:a", "libmp3lame", "-b:a", bitrate, out)
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover - depends on the ffmpeg build
        dst = os.path.splitext(out)[0] + ".wav"
        shutil.copy(wav, dst)
        return dst


def _mono(path: str) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


# ---------------------------------------------------------------- beats


def beat_grid(mix: np.ndarray, sig: dict, notes: dict) -> dict:
    """Beats, bar lines, meter and tempo (the Duet Arranger's beat tracker)."""
    from duet_arranger.audio.drums import prominence
    from duet_arranger.audio.rhythm import track_beats

    drums = sig.get("drums")
    prom = prominence(mix, drums) if drums is not None else 0.0
    harmonic = sum(sig[k] for k in ("piano", "guitar", "other", "bass") if k in sig) if sig else mix
    note_list = [(e.start, e.end, e.pitch, s) for s, evs in notes.items() for e in evs]
    bm = track_beats(mix, drums, sig.get("bass"), harmonic, prom, note_list)
    beats = [float(t) for t in bm.beats]
    m, ph = bm.beats_per_bar, bm.downbeat_phase
    bar_starts = ([0] if ph > 0 else []) + list(range(ph, len(beats), m))
    if bm.eighths:
        meter = bm.eighths
        per_felt = 3 if bm.eighths == "6/8" else 2
    elif bm.compound:
        meter, per_felt = f"{3 * m}/8", 1
    else:
        meter, per_felt = f"{m}/4", 1
    ibi = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    felt_bpm = 60.0 / (ibi * per_felt)
    dotted = meter.endswith("/8")
    return {
        "beats": [round(t, 4) for t in beats],
        "bars": [round(beats[i], 4) for i in bar_starts if i < len(beats)],
        "beats_per_bar": m,
        "meter": meter,
        "tempo": round(felt_bpm),
        "tempo_label": f"{'♩.' if dotted else '♩'} = {round(felt_bpm)}",
        "drum_prominence": round(float(prom), 3),
    }


# ---------------------------------------------------------------- chords


def frame_times(beats: list[float], duration: float) -> list[float]:
    """Chord frames: every tracked pulse, halved when pulses are slow."""
    if len(beats) < 2:
        return list(np.arange(0.0, duration, 0.5))
    beats = [b for b in beats if b >= 0]
    if len(beats) < 2:
        return list(np.arange(0.0, duration, 0.5))
    out: list[float] = [0.0] if beats[0] > 0.05 else []
    for a, b in zip(beats, beats[1:]):
        out.append(a)
        if b - a > 0.42:
            out.append((a + b) / 2)
    ibi = beats[-1] - beats[-2]
    t = beats[-1]
    while t < duration:
        out.append(t)
        t += ibi
    return out


def audio_chroma(y: np.ndarray, frames: list[float], duration: float) -> tuple[np.ndarray, np.ndarray]:
    """CQT chroma and RMS of a signal averaged over each frame."""
    import librosa

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=SR, hop_length=hop)
    edges = list(frames) + [duration]
    C = np.zeros((len(frames), 12))
    E = np.zeros(len(frames))
    for k in range(len(frames)):
        i0, i1 = np.searchsorted(times, [edges[k], edges[k + 1]])
        if i1 > i0:
            C[k] = chroma[:, i0:i1].mean(axis=1)
            E[k] = rms[min(i0, len(rms) - 1) : max(i0 + 1, min(i1, len(rms)))].mean()
    return C, E


def note_chroma(events: list, frames: list[float], duration: float, low_only: bool = False) -> np.ndarray:
    edges = list(frames) + [duration]
    C = np.zeros((len(frames), 12))
    starts = np.array(frames)
    for e in events:
        k0 = max(0, int(np.searchsorted(starts, e.start, side="right")) - 1)
        k = k0
        while k < len(frames) and edges[k] < e.end:
            ov = min(e.end, edges[k + 1]) - max(e.start, edges[k])
            if ov > 0:
                C[k, e.pitch % 12] += ov / max(1e-3, edges[k + 1] - edges[k]) * (0.5 + e.amplitude)
            k += 1
    return C


def pick_key(hist: np.ndarray, chords: list[tuple[float, float, int, str]]) -> tuple[int, str]:
    """Key from the notes' pitch classes and the chords: neighbouring keys
    share six of seven notes, so the chord the song rests on, opens and
    closes with, and how many of its chords belong to the scale, decide."""
    from duet_arranger.theory import MAJOR_PROFILE, MINOR_PROFILE, chord_pcs, scale_pcs

    total = sum(b - a for a, b, _, _ in chords) or 1.0
    best, best_score = (0, "major"), -9.0
    for tonic in range(12):
        for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            score = float(np.corrcoef(np.roll(prof, tonic), hist)[0, 1]) if hist.sum() > 0 else 0.0
            if chords:
                tq = ("maj", "maj7", "sus2", "sus4", "7") if mode == "major" else ("min", "m7")
                sc = set(scale_pcs(tonic, mode))
                home = sum(b - a for a, b, r, q in chords if r == tonic and q in tq) / total
                diat = sum(b - a for a, b, r, q in chords if set(chord_pcs(r, q)) <= sc) / total
                ends = sum(0.08 for c in (chords[0], chords[-1]) if c[2] == tonic and c[3] in tq)
                score += 0.6 * home + 0.5 * diat + ends
            if score > best_score:
                best, best_score = (tonic, mode), score
    return best


def raw_chords(chroma: np.ndarray, active: np.ndarray, frames: list[float], duration: float) -> list[tuple[float, float, int, str]]:
    """Chord roots and qualities without a key (to find the key from)."""
    from duet_arranger.theory import recognise_chords

    labels = recognise_chords(chroma + 1e-6, None, None)
    edges = list(frames) + [duration]
    out: list[list] = []
    for k, (r, q, _) in enumerate(labels):
        if not active[k]:
            continue
        if out and out[-1][2] == r and out[-1][3] == q and abs(out[-1][1] - edges[k]) < 1e-6:
            out[-1][1] = edges[k + 1]
        else:
            out.append([edges[k], edges[k + 1], r, q])
    return [tuple(c) for c in out if c[1] - c[0] >= 0.4]


def label_chords(chroma: np.ndarray, active: np.ndarray, frames: list[float], duration: float, key: tuple[int, str], flats: bool, bass: np.ndarray | None = None, min_len: float = 0.45) -> list[list]:
    """Viterbi chord labels per frame, merged into [start, end, name] segments
    (None where the instrument is silent)."""
    from duet_arranger.theory import chord_name, recognise_chords

    if len(frames) == 0:
        return []
    labels = recognise_chords(chroma + 1e-6, key, bass if bass is not None and bass.sum() > 0 else None)
    edges = list(frames) + [duration]
    segs: list[list] = []
    for k, (r, q, b) in enumerate(labels):
        name = chord_name(r, q, b, flats) if active[k] and chroma[k].sum() > 1e-6 else None
        if segs and segs[-1][2] == name:
            segs[-1][1] = edges[k + 1]
        else:
            segs.append([edges[k], edges[k + 1], name])
    # short chords are passing notes: they join the chord before (or after)
    merged: list[list] = []
    for s in segs:
        if merged and s[1] - s[0] < min_len and s[2] is not None:
            merged[-1][1] = s[1]
            continue
        if merged and merged[-1][2] == s[2]:
            merged[-1][1] = s[1]
            continue
        merged.append(s)
    return [[round(max(0.0, a), 3), round(b, 3), n] for a, b, n in merged if n is not None]


# ---------------------------------------------------------------- main


def analyse(src: str, workdir: str, progress: Progress = None, device: str | None = None, check_cancel: Callable[[], None] | None = None) -> dict:
    """Run the whole analysis into ``workdir``; returns the analysis dict."""
    from duet_arranger.audio.drums import detect_hits
    from duet_arranger.audio.separate import separate
    from duet_arranger.audio.transcribe import stem_is_silent, transcribe
    from duet_arranger.theory import prefers_flats

    def tick():
        if check_cancel:
            check_cancel()

    os.makedirs(workdir, exist_ok=True)
    mix_wav = os.path.join(workdir, "mix.wav")
    _report(progress, "decode", 0.1, "Reading the file")
    duration = decode(src, mix_wav)
    _report(progress, "decode", 1.0, f"{duration / 60:.0f}:{duration % 60:02.0f} of audio")
    tick()

    _report(progress, "separate", 0.0, "Loading the separation model")
    stems = separate(mix_wav, os.path.join(workdir, "stems"), device=device, progress=progress)
    _report(progress, "separate", 1.0, "Instruments separated")
    tick()

    mix = _mono(mix_wav)
    sig = {}
    for s, _, _ in TRACKS:
        p = stems.get(s)
        if p and os.path.exists(p):
            y = _mono(p)
            sig[s] = y[: len(mix)] if len(y) >= len(mix) else np.pad(y, (0, len(mix) - len(y)))
    rms = {s: float(np.sqrt(np.mean(y**2))) for s, y in sig.items()}
    loud = max([v for k, v in rms.items() if k != "drums"] or [1e-9])
    silent = {s for s in sig if stem_is_silent(stems[s]) or rms[s] < 0.04 * loud}

    notes: dict[str, list] = {}
    todo = [s for s in ("vocals", "piano", "other", "bass", "guitar") if s in sig and s not in silent]
    for k, s in enumerate(todo):
        _report(progress, "transcribe", k / max(1, len(todo)), f"Listening to the {dict((a, b) for a, b, _ in TRACKS)[s].lower()}")
        notes[s] = transcribe(stems[s], s)
        tick()
    _report(progress, "transcribe", 1.0, "Notes found")

    _report(progress, "rhythm", 0.2, "Finding the beat and the bar lines")
    grid = beat_grid(mix, sig, notes)
    _report(progress, "rhythm", 1.0, f"{grid['meter']} · {grid['tempo_label']}")
    tick()

    _report(progress, "chords", 0.1, "Naming the chords")
    hist = np.zeros(12)
    for s, evs in notes.items():
        if s == "drums":
            continue
        for e in evs:
            hist[e.pitch % 12] += (e.end - e.start) * (0.5 + e.amplitude) * (1.5 if s == "bass" else 1.0)
    frames = frame_times(grid["beats"], duration)
    beat_sec = float(np.median(np.diff(grid["beats"]))) if len(grid["beats"]) > 1 else 0.5
    if grid["meter"] in ("6/8", "3/4") and grid["beats_per_bar"] == 6:
        beat_sec *= 2  # pulses are eighth notes
    bass_c = note_chroma(notes.get("bass", []), frames, duration)
    combs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    song_c = np.zeros((len(frames), 12))
    song_active = np.zeros(len(frames), dtype=bool)
    for s in ("piano", "guitar", "other"):
        if s not in sig or s in silent:
            continue
        C, E = audio_chroma(sig[s], frames, duration)
        Cn = C / (C.max(axis=1, keepdims=True) + 1e-9)
        N = note_chroma(notes.get(s, []), frames, duration)
        Nn = N / (N.max(axis=1, keepdims=True) + 1e-9)
        # the guitar's strums transcribe poorly: its sound decides; elsewhere both
        comb = Cn * (1.0 if s == "guitar" else 0.6) + Nn * (0.3 if s == "guitar" else 0.8)
        ref = float(np.percentile(E[E > 0], 90)) if np.any(E > 0) else 1.0
        active = E > 0.12 * ref
        combs[s] = (comb, active)
        w = rms[s] / loud
        song_c += comb * w * active[:, None]
        song_active |= active
        tick()
    song_c = song_c + 0.5 * (bass_c / (bass_c.max(axis=1, keepdims=True) + 1e-9))
    tonic, mode = pick_key(hist, raw_chords(song_c, song_active, frames, duration))
    flats = prefers_flats(tonic, mode)
    per_track_chords = {s: label_chords(c, a, frames, duration, (tonic, mode), flats, min_len=0.9 * beat_sec) for s, (c, a) in combs.items()}
    _report(progress, "chords", 0.7, "Chords of the whole band")
    song_chords = label_chords(song_c, song_active, frames, duration, (tonic, mode), flats, bass=bass_c, min_len=1.4 * beat_sec)

    hits = []
    if "drums" in sig and "drums" not in silent:
        hits = [[round(float(t), 3), kind, round(float(st), 2)] for t, kind, st in detect_hits(sig["drums"])]
    _report(progress, "chords", 1.0, "Chords found")

    _report(progress, "encode", 0.0, "Preparing the tracks for playback")
    audio_dir = os.path.join(workdir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    original = encode(mix_wav, os.path.join(audio_dir, "original.mp3"), mono=False, bitrate="192k")
    tracks = []
    for k, (s, label, show) in enumerate(TRACKS):
        p = stems.get(s)
        if not p or not os.path.exists(p):
            continue
        out = encode(p, os.path.join(audio_dir, f"{s}.mp3"), mono=True, bitrate="128k")
        _report(progress, "encode", (k + 1) / len(TRACKS), f"Prepared the {label.lower()}")
        evs = notes.get(s, []) if "notes" in show else []
        tracks.append(
            {
                "id": s,
                "label": label,
                "show": show,
                "audio": "audio/" + os.path.basename(out),
                "silent": s in silent,
                "level": round(rms.get(s, 0.0) / loud, 3) if loud else 0.0,
                "notes": [[round(e.start, 3), round(e.end, 3), int(e.pitch), round(float(e.amplitude), 2)] for e in sorted(evs, key=lambda e: (e.start, e.pitch))],
                "chords": per_track_chords.get(s, []),
                "hits": hits if s == "drums" else [],
            }
        )
        tick()
    from duet_arranger.theory import pc_name

    result = {
        "version": ANALYSIS_VERSION,
        "duration": round(duration, 3),
        "original": "audio/" + os.path.basename(original),
        "key": f"{pc_name(tonic, flats)} {mode}",
        "flats": flats,
        **grid,
        "chords": song_chords,
        "tracks": tracks,
    }
    with open(os.path.join(workdir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    # the separated WAVs and the decoded mix are large; the MP3s are what the player needs
    try:
        os.remove(mix_wav)
    except OSError:
        pass
    shutil.rmtree(os.path.join(workdir, "stems"), ignore_errors=True)
    _report(progress, "encode", 1.0, "Ready")
    return result
