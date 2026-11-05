"""Audio file -> Song: separate, transcribe, find the beat, analyse."""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from ..analysis import finish_analysis
from ..model import TPQ, Hit, Note, Song
from ..progress import report
from ..quantize import Grid, detect_triplet_beats, snap_notes
from .rhythm import SR, BeatMap, track_beats
from .separate import STEMS, load_audio, separate
from .transcribe import NoteEvent, stem_is_silent, transcribe

log = logging.getLogger(__name__)

TRANSCRIBE = ("vocals", "bass", "piano", "guitar", "other")
# Bump when the analysis changes in ways a saved Song should not skip
# (2: eighth-note grid, steady tempo, vocal-bleed removal; 3: beat-snapped
# chord changes, stronger start-on-beat-1 downbeat prior; 4: 6/8 and 3/4 read
# from eighth-note pulses, leakage cleaning, chord-based key choice).
ANALYSIS_VERSION = 4


def _mono(path: str) -> np.ndarray:
    data, _ = load_audio(path, SR, mono=True)
    return data[0]


def analyze_audio(path: str, workdir: str, title: str | None = None, artist: str = "", device: str | None = None, stems: dict[str, str] | None = None, progress=None) -> Song:
    os.makedirs(workdir, exist_ok=True)
    stem_dir = os.path.join(workdir, "stems")
    report(progress, "separate", 0.0, "Loading the separation model")
    stems = stems or separate(path, stem_dir, device=device, progress=progress)
    report(progress, "separate", 1.0, "Stems ready")
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
    for k, s in enumerate(TRANSCRIBE):
        p = stems.get(s)
        report(progress, "transcribe", k / len(TRANSCRIBE), f"Transcribing {s}")
        if not p or not os.path.exists(p):
            continue
        if stem_is_silent(p):
            log.info("Stem %s is silent; skipping", s)
            continue
        log.info("Transcribing %s …", s)
        events[s] = transcribe(p, s)
    note_list = [(e.start, e.end, e.pitch, s) for s, evs in events.items() for e in evs]
    report(progress, "transcribe", 1.0, "Notes transcribed")
    report(progress, "rhythm", 0.1, "Tracking the beat")
    beatmap = track_beats(mix, drums, sig.get("bass"), harmonic, prom, note_list)
    song = Song(title=title or os.path.splitext(os.path.basename(path))[0], artist=artist)
    song.bars = beatmap.bars()
    report(progress, "rhythm", 1.0, f"{len(song.bars)} bars found")
    report(progress, "analyse", 0.1, "Placing notes on the bar grid")
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
    song.parts = clean_leakage(parts)
    song.melody = list(song.parts.get("vocals", []))
    song.bass = list(song.parts.get("bass", []))
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
    if "vocals" in sig and song.parts.get("vocals"):
        drop_vocal_bleed(song, sig["vocals"], beatmap)
    extra = _chroma_frames(harmonic, beatmap, song)
    finish_analysis(song, rms, extra, prom)
    trim_silence(song)
    report(progress, "analyse", 1.0, "Key, chords and sections found")
    song.notes_info = {
        "beats": len(beatmap.beats),
        "beats_per_bar": beatmap.beats_per_bar,
        "compound": beatmap.compound,
        "stems": {s: len(v) for s, v in parts.items()},
        "stem_rms": {s: round(float(np.sqrt(np.mean(v**2))), 5) for s, v in sig.items()},
        "hits": len(song.hits),
        "drum_prominence": round(prom, 3),
    }
    with open(os.path.join(workdir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(song_summary(song), f, indent=1)
    return song


def clean_leakage(parts: dict[str, list[Note]]) -> dict[str, list[Note]]:
    """Remove transcription debris from the harmony stems.

    * bass leaking into the piano/guitar/other stems: quiet low notes that
      match the bass note sounding at that moment;
    * blips: very short, quiet notes standing alone (no chord around them).
    """
    from ..arrange.lines import LineIndex, bassline

    bass = LineIndex(bassline(parts.get("bass", [])))
    out = dict(parts)
    for name, notes in parts.items():
        if name in ("vocals", "bass", "drums"):
            continue
        onsets: dict[int, int] = {}
        for n in notes:
            onsets[n.start] = onsets.get(n.start, 0) + 1
        kept = []
        for n in notes:
            b = bass.at(n.start) or bass.next_after(n.start - 1)
            if n.pitch < 52 and n.velocity < 0.8 and b is not None and abs(b.start - n.start) <= TPQ * 2 and (n.pitch - b.pitch) % 12 == 0:
                continue
            if n.end - n.start <= 6 and n.velocity < 0.7 and onsets.get(n.start, 0) == 1:
                continue
            kept.append(n)
        out[name] = kept
    return out


def drop_vocal_bleed(song: Song, vocals: np.ndarray, beatmap: BeatMap, floor: float = 0.14) -> None:
    """Forget "vocal" notes in bars where nobody sings.

    Separation leaves faint piano or reverb in the vocal stem; transcribed, it
    becomes a fake melody (often in an instrumental intro). Bars whose vocal
    energy is below ``floor`` of the stem's loud bars keep no vocal notes.
    """
    levels = []
    for b in song.bars:
        t0 = _tick_seconds(beatmap, b.start)
        t1 = _tick_seconds(beatmap, b.end)
        seg = vocals[int(t0 * SR) : max(int(t0 * SR) + 1, int(t1 * SR))]
        levels.append(float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0)
    if not levels:
        return
    loud = float(np.percentile(levels, 85)) or 1e-9
    quiet = {i for i, v in enumerate(levels) if v < floor * loud}
    kept = [n for n in song.parts["vocals"] if song.bar_at(n.start) not in quiet]
    log.info("Vocal activity: %d of %d bars sung, %d bleed notes dropped", len(levels) - len(quiet), len(levels), len(song.parts["vocals"]) - len(kept))
    song.parts["vocals"] = kept
    song.melody = list(kept)


def _tick_seconds(bm: BeatMap, tick: float) -> float:
    tpb = bm.pulse_ticks
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
    """Drop empty bars before the music starts and after it ends.

    "Empty" means no real music: reverb tails, stray drum-stem blips and
    faint notes do not count. A short pickup bar is kept only when it holds
    a fair share of what the next bar holds (a real anacrusis)."""
    used = [n.start for ns in song.parts.values() for n in ns] + [h.start for h in song.hits]
    if not used or not song.bars:
        return
    activity = [0.0] * len(song.bars)
    for name, ns in song.parts.items():
        for n in ns:
            if n.velocity >= 0.35:
                activity[song.bar_at(n.start)] += n.velocity
    for h in song.hits:
        if h.velocity >= 0.4:
            activity[song.bar_at(h.start)] += 0.5 * h.velocity
    busy = sorted(a for a in activity if a > 0)
    floor = 0.15 * busy[len(busy) // 2] if busy else 0.0
    first = next((i for i, a in enumerate(activity) if a > floor), song.bar_at(min(used)))
    if first + 1 < len(song.bars) and song.bars[first].length < song.bars[first + 1].length and activity[first] < 0.35 * activity[first + 1]:
        first += 1
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
