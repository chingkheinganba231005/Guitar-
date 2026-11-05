"""Symbolic analysis shared by the MIDI and audio paths: key, chords,
sections, energy and drum prominence, all on the bar grid."""

from __future__ import annotations

import numpy as np

from .model import TPQ, Chord, Note, Section, Song
from .theory import detect_key, recognise_chords

HARMONIC_SOURCES = ("piano", "guitar", "other", "keys", "synth", "strings", "bass", "vocals", "lead")


def pitch_class_histogram(notes: list[Note]) -> np.ndarray:
    h = np.zeros(12)
    for n in notes:
        h[n.pitch % 12] += (n.end - n.start) * (0.5 + n.velocity)
    return h


def estimate_key(song: Song) -> None:
    notes = [n for name, ns in song.parts.items() if name != "drums" for n in ns] + song.melody + song.bass
    tonic, mode, _ = detect_key(pitch_class_histogram(notes))
    song.key_tonic, song.key_mode = tonic, mode


def chord_frames(song: Song, step: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Duration-weighted chroma (harmony) and bass chroma per ``step`` ticks."""
    end = song.end_tick
    n = max(1, (end + step - 1) // step)
    chroma = np.zeros((n, 12))
    bass = np.zeros((n, 12))
    starts = [i * step for i in range(n)]
    harm = [x for name, ns in song.parts.items() if name not in ("drums", "vocals") for x in ns]
    for x in harm:
        a = max(0, x.start // step)
        b = min(n, max(a + 1, (x.end + step - 1) // step))
        w = 0.4 + x.velocity
        for i in range(a, b):
            lo = max(x.start, i * step)
            hi = min(x.end, (i + 1) * step)
            if hi > lo:
                chroma[i, x.pitch % 12] += w * (hi - lo) / step
    for x in song.bass:
        a = max(0, x.start // step)
        b = min(n, max(a + 1, (x.end + step - 1) // step))
        for i in range(a, b):
            lo = max(x.start, i * step)
            hi = min(x.end, (i + 1) * step)
            if hi > lo:
                bass[i, x.pitch % 12] += (hi - lo) / step
                chroma[i, x.pitch % 12] += 0.6 * (hi - lo) / step
    # The melody colours the harmony a little (helps when the accompaniment is thin).
    for x in song.melody:
        a = max(0, x.start // step)
        b = min(n, max(a + 1, (x.end + step - 1) // step))
        for i in range(a, b):
            chroma[i, x.pitch % 12] += 0.25
    return chroma, bass, starts


def estimate_chords(song: Song, extra_chroma: np.ndarray | None = None) -> None:
    """Label chords per half beat, then merge runs into chord segments."""
    if not song.bars:
        return
    step = TPQ // 2
    chroma, bass, starts = chord_frames(song, step)
    if extra_chroma is not None and len(extra_chroma) == len(chroma):
        chroma = chroma + extra_chroma
    labels = recognise_chords(chroma, (song.key_tonic, song.key_mode), bass if bass.sum() > 0 else None)
    chords: list[Chord] = []
    for i, (r, q, b) in enumerate(labels):
        if chroma[i].sum() <= 1e-6:
            if chords:
                chords[-1].end = starts[i] + step
            continue
        if chords and chords[-1].root == r and chords[-1].quality == q and chords[-1].bass == b:
            chords[-1].end = starts[i] + step
        else:
            chords.append(Chord(starts[i], starts[i] + step, r, q, b))
    # Very short chords are usually passing tones: absorb them.
    merged: list[Chord] = []
    for c in chords:
        if merged and c.end - c.start < TPQ and c.start - merged[-1].end <= step:
            merged[-1].end = c.end
            continue
        merged.append(c)
    # Chord changes happen on beats, most often on the barline: an estimate a
    # little early or late (an anticipating melody note, a ringing bass) moves.
    bar_starts = [b.start for b in song.bars]
    beat_starts = song.beat_starts()
    for c in merged[1:]:
        near_bar = min(bar_starts, key=lambda t: abs(t - c.start))
        near_beat = min(beat_starts, key=lambda t: abs(t - c.start)) if beat_starts else c.start
        if abs(near_bar - c.start) <= TPQ // 2:
            c.start = near_bar
        elif abs(near_beat - c.start) <= TPQ // 2:
            c.start = near_beat
    snapped: list[Chord] = []
    for c in merged:
        if snapped and c.start <= snapped[-1].start:
            continue
        if snapped and (c.root, c.quality, c.bass) == (snapped[-1].root, snapped[-1].quality, snapped[-1].bass):
            continue
        snapped.append(c)
    merged = snapped
    for a, b in zip(merged, merged[1:]):
        a.end = b.start
    if merged:
        merged[-1].end = max(merged[-1].end, song.end_tick)
    song.chords = merged


def estimate_energy(song: Song, per_bar_rms: list[float] | None = None) -> None:
    if per_bar_rms:
        a = np.asarray(per_bar_rms, dtype=float)
    else:
        a = np.zeros(len(song.bars))
        all_notes = [x for ns in song.parts.values() for x in ns] + song.melody
        for x in all_notes:
            bi = song.bar_at(x.start)
            a[bi] += 0.3 + x.velocity
        for h in song.hits:
            a[song.bar_at(h.start)] += 0.5 * h.velocity
    if len(a) == 0:
        song.energy = []
        return
    lo, hi = np.percentile(a, 5), np.percentile(a, 95)
    if hi - lo < 1e-9:
        song.energy = [0.5] * len(a)
        return
    song.energy = [float(np.clip((v - lo) / (hi - lo), 0, 1)) for v in a]


def estimate_sections(song: Song) -> None:
    """Self-similarity segmentation of bars into labelled sections."""
    nb = len(song.bars)
    if nb == 0:
        return
    if nb < 8:
        song.sections = [Section(0, "Song")]
        return
    feats = []
    mel_bars = set(song.bar_at(n.start) for n in song.melody)
    for bi, b in enumerate(song.bars):
        v = np.zeros(12)
        for c in song.chords:
            ov = min(c.end, b.end) - max(c.start, b.start)
            if ov > 0:
                for p in c.pcs:
                    v[p] += ov
        m = np.zeros(12)
        for n in song.melody:
            if b.start <= n.start < b.end:
                m[n.pitch % 12] += n.end - n.start
        f = np.concatenate([v / (np.linalg.norm(v) + 1e-9), 0.7 * m / (np.linalg.norm(m) + 1e-9), [song.energy[bi] if song.energy else 0.5]])
        feats.append(f)
    F = np.array(feats)
    # novelty across 4-bar windows, boundaries on multiples of 4 bars where possible
    w = 4
    nov = np.zeros(nb)
    for i in range(w, nb - w + 1):
        a = F[i - w : i].mean(axis=0)
        b = F[i : i + w].mean(axis=0)
        nov[i] = np.linalg.norm(a - b)
    for i in range(1, nb):
        if (i in mel_bars) != ((i - 1) in mel_bars):
            nov[i] += 0.35  # singing starts or stops
    bounds = [0]
    thresh = np.percentile(nov[nov > 0], 70) if np.any(nov > 0) else 1.0
    for i in range(2, nb - 1):
        if nov[i] >= thresh and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1] and i - bounds[-1] >= 4:
            bounds.append(i)
    segs = [(a, b) for a, b in zip(bounds, bounds[1:] + [nb])]
    # cluster segments by mean feature
    means = [F[a:b].mean(axis=0) for a, b in segs]
    labels: list[int] = []
    protos: list[np.ndarray] = []
    for m in means:
        for k, p in enumerate(protos):
            if np.linalg.norm(m - p) < 0.55:
                labels.append(k)
                break
        else:
            protos.append(m)
            labels.append(len(protos) - 1)
    energy = [float(np.mean(song.energy[a:b])) if song.energy else 0.5 for a, b in segs]
    has_vocals = [any(i in mel_bars for i in range(a, b)) for a, b in segs]
    counts = {k: labels.count(k) for k in set(labels)}
    vocal_clusters = [k for k in counts if any(has_vocals[i] for i in range(len(segs)) if labels[i] == k)]
    chorus = None
    if vocal_clusters:
        chorus = max(vocal_clusters, key=lambda k: (counts[k] > 1, np.mean([energy[i] for i in range(len(segs)) if labels[i] == k])))
    names = {}
    out = []
    for i, ((a, b), k) in enumerate(zip(segs, labels)):
        if not has_vocals[i]:
            name = "Intro" if i == 0 else ("Outro" if i == len(segs) - 1 else "Interlude")
        elif k == chorus:
            name = "Chorus"
        else:
            if k not in names:
                if not names:
                    names[k] = "Verse"
                elif counts[k] == 1 and i > len(segs) / 2:
                    names[k] = "Bridge"
                elif len(names) == 1:
                    names[k] = "Pre-chorus"
                else:
                    names[k] = f"Part {chr(64 + len(names))}"
            name = names[k]
        out.append(Section(a, name))
    song.sections = out


def drum_prominence_from_hits(song: Song) -> float:
    if not song.bars:
        return 0.0
    per_bar = len([h for h in song.hits if h.kind in ("kick", "snare")]) / len(song.bars)
    return float(min(1.0, per_bar / 4.0))


def finish_analysis(song: Song, per_bar_rms: list[float] | None = None, extra_chroma=None, prominence: float | None = None) -> Song:
    estimate_key(song)
    estimate_energy(song, per_bar_rms)
    if not song.chords:
        estimate_chords(song, extra_chroma)
    if not song.sections:
        estimate_sections(song)
    if prominence is None:
        prominence = drum_prominence_from_hits(song)
    song.drum_prominence = prominence
    snares = [h for h in song.hits if h.kind == "snare"]
    song.percussive = prominence >= 0.35 and len(snares) >= max(4, len(song.bars) // 2)
    return song
