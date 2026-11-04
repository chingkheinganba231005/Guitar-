"""Audio file -> Song: separate, transcribe, find the beat, analyse."""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from ..analysis import finish_analysis
from ..model import TPQ, Hit, Note, Song
from ..quantize import Grid, detect_triplet_beats, snap_notes
from .rhythm import SR, BeatMap, track_beats
from .separate import STEMS, load_audio, separate
from .transcribe import NoteEvent, stem_is_silent, transcribe

log = logging.getLogger(__name__)

TRANSCRIBE = ("vocals", "bass", "piano", "guitar", "other")


def _mono(path: str) -> np.ndarray:
    data, _ = load_audio(path, SR, mono=True)
    return data[0]


def analyze_audio(path: str, workdir: str, title: str | None = None, artist: str = "", device: str | None = None, stems: dict[str, str] | None = None) -> Song:
    os.makedirs(workdir, exist_ok=True)
    stem_dir = os.path.join(workdir, "stems")
    stems = stems or separate(path, stem_dir, device=device)
    mix = _mono(path)
    sig = {s: _mono(stems[s]) for s in STEMS if s in stems and os.path.exists(stems[s])}
    n = len(mix)
    for k in sig:
        sig[k] = sig[k][:n] if len(sig[k]) >= n else np.pad(sig[k], (0, n - len(sig[k])))
    from .drums import detect_hits, prominence

    drums = sig.get("drums")
    prom = prominence(mix, drums) if drums is not None else 0.0
    harmonic = sum(sig[k] for k in ("piano", "guitar", "other", "bass") if k in sig) if sig else mix
    # transcription per stem (in seconds; independent of the beat grid)
    events: dict[str, list[NoteEvent]] = {}
    for s in TRANSCRIBE:
        p = stems.get(s)
        if not p or not os.path.exists(p):
            continue
        if stem_is_silent(p):
            log.info("Stem %s is silent; skipping", s)
            continue
        log.info("Transcribing %s …", s)
        events[s] = transcribe(p, s)
    note_list = [(e.start, e.end, e.pitch, s) for s, evs in events.items() for e in evs]
    beatmap = track_beats(mix, drums, sig.get("bass"), harmonic, prom, note_list)
    song = Song(title=title or os.path.splitext(os.path.basename(path))[0], artist=artist)
    song.bars = beatmap.bars()
    onsets = []
    parts: dict[str, list[Note]] = {}
    for s, evs in events.items():
        ns = []
        for e in evs:
            st = beatmap.to_ticks(e.start)
            en = beatmap.to_ticks(e.end)
            ns.append(Note(st, max(en, st + 1), e.pitch, velocity=float(np.clip(e.amplitude * 1.6, 0.05, 1.0)), source=s, tags=set(e.tags)))
            onsets.append(st)
        parts[s] = ns
    song.triplet_beats = detect_triplet_beats(song, onsets)
    for s in list(parts):
        parts[s] = snap_notes(parts[s], song)
    song.parts = parts
    song.melody = list(parts.get("vocals", []))
    song.bass = list(parts.get("bass", []))
    grid = Grid(song)
    if drums is not None:
        for t, kind, strength in detect_hits(drums):
            tick = grid.snap(beatmap.to_ticks(t))
            if tick < song.end_tick:
                song.hits.append(Hit(tick, kind, strength))
    # bar energy from the full mix
    rms = []
    for b in song.bars:
        t0 = _tick_seconds(beatmap, b.start)
        t1 = _tick_seconds(beatmap, b.end)
        seg = mix[int(t0 * SR) : max(int(t0 * SR) + 1, int(t1 * SR))]
        rms.append(float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0)
    extra = _chroma_frames(harmonic, beatmap, song)
    finish_analysis(song, rms, extra, prom)
    trim_silence(song)
    song.notes_info = {
        "beats": len(beatmap.beats),
        "beats_per_bar": beatmap.beats_per_bar,
        "compound": beatmap.compound,
        "stems": {s: len(v) for s, v in parts.items()},
        "hits": len(song.hits),
        "drum_prominence": round(prom, 3),
    }
    with open(os.path.join(workdir, "analysis.json"), "w") as f:
        json.dump(song_summary(song), f, indent=1)
    return song


def _tick_seconds(bm: BeatMap, tick: float) -> float:
    tpb = bm.ticks_per_beat
    i = tick / tpb
    j = int(np.floor(i))
    b = bm.beats
    if j >= len(b) - 1:
        return float(b[-1] + (i - (len(b) - 1)) * (b[-1] - b[-2]))
    return float(b[j] + (i - j) * (b[j + 1] - b[j]))


def _chroma_frames(y: np.ndarray, bm: BeatMap, song: Song) -> np.ndarray | None:
    """CQT chroma of the harmonic stems on the half-beat chord frames."""
    try:
        import librosa

        hop = 512
        chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=hop)
        step = TPQ // 2
        nfr = max(1, (song.end_tick + step - 1) // step)
        out = np.zeros((nfr, 12))
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=SR, hop_length=hop)
        for k in range(nfr):
            a = _tick_seconds(bm, k * step)
            b = _tick_seconds(bm, (k + 1) * step)
            i0, i1 = np.searchsorted(times, [a, b])
            if i1 > i0:
                out[k] = chroma[:, i0:i1].mean(axis=1)
        return out * 0.8
    except Exception as err:  # pragma: no cover - chroma is an optional refinement
        log.warning("Chroma unavailable: %s", err)
        return None


def trim_silence(song: Song) -> None:
    """Drop empty bars before the music starts and after it ends."""
    used = [n.start for ns in song.parts.values() for n in ns] + [h.start for h in song.hits]
    if not used or not song.bars:
        return
    first = song.bar_at(min(used))
    last_tick = max([n.end for ns in song.parts.values() for n in ns] + [h.start + 1 for h in song.hits])
    last = song.bar_at(max(0, last_tick - 1))
    if first == 0 and last == len(song.bars) - 1:
        return
    offset = song.bars[first].start
    keep = song.bars[first : last + 1]
    for b in keep:
        b.start -= offset

    def shift_notes(ns):
        out = []
        for n in ns:
            if n.start < offset or n.start >= keep[-1].end + offset:
                continue
            n.start -= offset
            n.end = min(n.end - offset, keep[-1].end)
            out.append(n)
        return out

    for k in list(song.parts):
        song.parts[k] = shift_notes(song.parts[k])
    song.melody = list(song.parts.get("vocals", []))
    song.bass = list(song.parts.get("bass", []))
    song.hits = [h for h in song.hits if offset <= h.start < keep[-1].end + offset]
    for h in song.hits:
        h.start -= offset
    for c in song.chords:
        c.start = max(0, c.start - offset)
        c.end = max(0, c.end - offset)
    song.chords = [c for c in song.chords if c.end > c.start and c.start < keep[-1].end]
    for s in song.sections:
        s.start_bar = max(0, s.start_bar - first)
    song.sections = [s for i, s in enumerate(song.sections) if s.start_bar < len(keep) and (i == 0 or s.start_bar != song.sections[i - 1].start_bar)]
    song.energy = song.energy[first : last + 1]
    grid_beats = song.bars[0].ticks_per_beat
    song.triplet_beats = {i - (offset // grid_beats) for i in song.triplet_beats if i >= offset // grid_beats}
    song.bars = keep


def song_summary(song: Song) -> dict:
    from ..theory import pc_name, prefers_flats

    flats = prefers_flats(song.key_tonic, song.key_mode)
    return {
        "title": song.title,
        "key": f"{pc_name(song.key_tonic, flats)} {song.key_mode}",
        "bars": len(song.bars),
        "meter": list(song.bars[0].meter) if song.bars else None,
        "tempo": round(song.bars[0].tempo, 1) if song.bars else None,
        "percussive": song.percussive,
        "drum_prominence": round(song.drum_prominence, 3),
        "sections": [{"bar": s.start_bar + 1, "label": s.label} for s in song.sections],
        "chords": [{"bar": song.bar_at(c.start) + 1, "name": c.name(flats)} for c in song.chords[:400]],
        "melody_notes": len(song.melody),
        "bass_notes": len(song.bass),
        "parts": {k: len(v) for k, v in song.parts.items()},
        **song.notes_info,
    }
