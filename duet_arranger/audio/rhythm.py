"""Beats, meter, downbeats and the seconds -> ticks map."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..model import TPQ, Bar

log = logging.getLogger(__name__)
SR = 22050
HOP = 512


@dataclass
class BeatMap:
    beats: np.ndarray  # seconds of every felt beat
    beats_per_bar: int
    downbeat_phase: int
    compound: bool

    @property
    def ticks_per_beat(self) -> int:
        return int(TPQ * 1.5) if self.compound else TPQ

    def bars(self) -> list[Bar]:
        """Bars on the tick grid; beat 0 is tick 0, a pickup bar covers beats before the first downbeat."""
        tpb = self.ticks_per_beat
        b = self.beats
        m = self.beats_per_bar
        out: list[Bar] = []

        def tempo_of(i0: int, i1: int) -> float:
            seg = b[i0 : min(i1 + 1, len(b))]
            ibi = float(np.median(np.diff(seg))) if len(seg) >= 2 else float(np.median(np.diff(b)))
            return float(np.clip(60.0 / ibi * (1.5 if self.compound else 1.0), 20, 300))

        ph = self.downbeat_phase
        if ph > 0:
            if self.compound:
                out.append(Bar(0, 3 * ph, 8, tempo_of(0, ph), tpb))
            else:
                out.append(Bar(0, ph, 4, tempo_of(0, ph), tpb))
        for k in range(ph, len(b) - 1, m):
            tick = k * tpb
            if self.compound:
                out.append(Bar(tick, 3 * m, 8, tempo_of(k, k + m), tpb))
            else:
                out.append(Bar(tick, m, 4, tempo_of(k, k + m), tpb))
        return out

    def to_ticks(self, t: float) -> float:
        """Map seconds to ticks by linear interpolation between beats (beat i = tick i*tpb)."""
        b = self.beats
        tpb = self.ticks_per_beat
        if t <= b[0]:
            return 0.0
        if t >= b[-1]:
            ibi = b[-1] - b[-2]
            return (len(b) - 1 + (t - b[-1]) / ibi) * tpb
        j = int(np.searchsorted(b, t) - 1)
        frac = (t - b[j]) / (b[j + 1] - b[j])
        return (j + frac) * tpb


def onset_envelope(y: np.ndarray, sr: int = SR, fmin: float | None = None, fmax: float | None = None, detrend: bool = True) -> np.ndarray:
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = np.ones_like(freqs, dtype=bool)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax
    S = librosa.amplitude_to_db(S[mask] + 1e-10, ref=np.max)
    flux = np.maximum(0.0, np.diff(S, axis=1)).mean(axis=0)
    flux = np.concatenate([[0.0], flux])
    if detrend:
        flux -= np.convolve(flux, np.ones(16) / 16, mode="same")
    return np.maximum(flux, 0.0)


def band_energy_at(y: np.ndarray, times: np.ndarray, fmin: float, fmax: float, win: float = 0.07) -> np.ndarray:
    """Linear spectral energy in [fmin, fmax] just after each time (onset strength proxy)."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    band = S[(freqs >= fmin) & (freqs <= fmax)].sum(axis=0)
    fr = librosa.time_to_frames(np.maximum(times, 0), sr=SR, hop_length=HOP)
    w = max(1, int(win * SR / HOP))
    out = np.array([band[max(0, f - 1) : f + w].max() if f < len(band) else 0.0 for f in fr])
    return out


def note_change_features(times: np.ndarray, notes: list[tuple[float, float, int, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Harmony change and bass change at each beat, from transcribed notes.

    ``notes``: (start_s, end_s, pitch, stem)."""
    n = len(times)
    hist = np.zeros((n, 12))
    bass_pc = -np.ones(n, dtype=int)
    bass_low = np.full(n, 999)
    for st, en, p, stem in notes:
        if stem == "drums":
            continue
        i0 = max(0, int(np.searchsorted(times, st, side="right") - 1))
        i1 = int(np.searchsorted(times, en, side="left"))
        for i in range(i0, min(n, max(i0 + 1, i1))):
            a = times[i]
            b = times[i + 1] if i + 1 < n else a + 0.5
            ov = min(en, b) - max(st, a)
            if ov > 0:
                hist[i, p % 12] += ov * (1.5 if stem == "bass" else 1.0)
                if stem == "bass" and p < bass_low[i]:
                    bass_low[i] = p
                    bass_pc[i] = p % 12
    hist = hist / (np.linalg.norm(hist, axis=1, keepdims=True) + 1e-9)
    harm = np.zeros(n)
    harm[1:] = 1 - (hist[1:] * hist[:-1]).sum(axis=1)
    bch = np.zeros(n)
    for i in range(1, n):
        if bass_pc[i] >= 0 and bass_pc[i - 1] >= 0 and bass_pc[i] != bass_pc[i - 1]:
            bch[i] = 1.0
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9) if v.std() > 0 else v * 0
    return z(harm), z(bch)


def track_beats(mix: np.ndarray, drums: np.ndarray | None, bass: np.ndarray | None, harmonic: np.ndarray | None, percussive_hint: float, notes: list | None = None) -> BeatMap:
    """Beat tracking + meter/downbeat selection.

    ``mix`` etc. are mono signals at 22.05 kHz. The meter (3 or 4 beats per
    bar) and downbeat phase maximise accents from kick/bass onsets and
    chord changes on the first beat of each bar.
    """
    import librosa

    env = onset_envelope(mix)
    if drums is not None and percussive_hint > 0.25:
        env = env + 1.5 * onset_envelope(drums)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, units="frames", tightness=110)
    beats = np.asarray(beats, dtype=int)
    if len(beats) < 8:
        # Rubato or near-silent: fall back to a steady grid from the global tempo.
        bpm = float(np.atleast_1d(tempo)[0]) or 90.0
        step = 60.0 / bpm
        dur = len(mix) / SR
        times = np.arange(0.0, dur + step, step)
    else:
        times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    low_env = onset_envelope(drums, fmax=160) if drums is not None else (onset_envelope(bass, fmax=300) if bass is not None else env)
    times = fix_tempo_octave(np.asarray(times, dtype=float), low_env)
    if notes:
        times = harmonic_rhythm_octave(times, notes)
    times = _extend(times, len(mix) / SR)
    # accent features per beat
    frames = librosa.time_to_frames(times, sr=SR, hop_length=HOP)
    frames = np.clip(frames, 0, len(env) - 1)

    def at_beats(e):
        if e is None:
            return np.zeros(len(times))
        out = np.zeros(len(times))
        for k, f in enumerate(frames):
            out[k] = e[max(0, f - 2) : f + 3].max() if len(e) else 0.0
        s = out.std()
        return (out - out.mean()) / s if s > 0 else out * 0

    def z(v):
        v = np.log1p(v / (np.median(v) + 1e-9)) if v.max() > 0 else v
        return (v - v.mean()) / (v.std() + 1e-9) if v.std() > 0 else v * 0

    low = z(band_energy_at(drums, times, 30, 120)) if drums is not None else np.zeros(len(times))
    bass_acc = at_beats(onset_envelope(bass, fmax=300, detrend=False) if bass is not None else None)
    chord_change = np.zeros(len(times))
    if harmonic is not None:
        chroma = librosa.feature.chroma_cqt(y=harmonic, sr=SR, hop_length=HOP)
        bounds = list(frames) + [chroma.shape[1]]
        sync = np.zeros((12, len(times)))
        for j in range(len(times)):
            a, b = bounds[j], max(bounds[j] + 1, bounds[j + 1])
            sync[:, j] = np.median(chroma[:, a:b], axis=1) if a < chroma.shape[1] else 0.0
        sync = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-9)
        diff = np.zeros(len(times))
        diff[1:] = 1 - (sync[:, 1:] * sync[:, :-1]).sum(axis=0)
        chord_change = (diff - diff.mean()) / (diff.std() + 1e-9)
    accent = 1.0 * low + 0.8 * bass_acc + 1.2 * chord_change
    if notes:
        harm_change, bass_change = note_change_features(times, notes)
        accent = accent + 1.6 * harm_change + 0.5 * bass_change
    # the first clearly audible onset usually falls on a downbeat
    first_beat = None
    if len(env):
        thr = 0.3 * env.max()
        idx = np.nonzero(env > thr)[0]
        if len(idx):
            t_first = librosa.frames_to_time(idx[0], sr=SR, hop_length=HOP)
            first_beat = int(np.argmin(np.abs(times - t_first)))
    compound = _is_compound(mix, times)
    best = (-1e9, 4, 0)
    for m in (4, 3, 2):
        for ph in range(m):
            sel = accent[ph::m]
            rest = np.delete(accent, np.arange(ph, len(accent), m))
            if len(sel) < 2 or len(rest) < 2:
                continue
            score = sel.mean() - rest.mean()
            if m == 4:
                score += 0.12  # prior: most popular music is in 4
            if m == 2:
                score -= 0.15
            if first_beat is not None and (first_beat - ph) % m == 0:
                score += 0.18
            if score > best[0]:
                best = (score, m, ph)
    _, m, ph = best
    if compound and m == 3:
        m = 2
    log.info("Beats: %d, %s, %d per bar, phase %d", len(times), "compound" if compound else "simple", m, ph)
    return BeatMap(np.asarray(times), m, ph, compound)


def fix_tempo_octave(times: np.ndarray, low_env: np.ndarray, lo: float = 62.0, hi: float = 150.0) -> np.ndarray:
    """Undo double/half-time beat tracking: keep the tempo in [lo, hi] BPM.

    When halving, keep the beat phase where kick/bass onsets land."""
    import librosa

    if len(times) < 4:
        return times
    for _ in range(3):
        bpm = 60.0 / float(np.median(np.diff(times)))
        if bpm > hi:
            frames = np.clip(librosa.time_to_frames(times, sr=SR, hop_length=HOP), 0, len(low_env) - 1)
            strength = np.array([low_env[max(0, f - 2) : f + 3].max() for f in frames])
            ph = 0 if strength[0::2].mean() >= strength[1::2].mean() else 1
            times = times[ph::2]
        elif bpm < lo:
            mids = (times[:-1] + times[1:]) / 2
            times = np.sort(np.concatenate([times, mids]))
        else:
            break
    return times


def harmonic_rhythm_octave(times: np.ndarray, notes: list) -> np.ndarray:
    """Halve a double-time beat grid when chords only change every ~8 beats.

    Pop harmony usually moves every 2–4 beats; a tracker locked to eighth
    notes sees changes every 8 "beats" instead."""
    if len(times) < 16:
        return times
    bpm = 60.0 / float(np.median(np.diff(times)))
    harm, _ = note_change_features(times, notes)
    changes = np.nonzero(harm > 0.8)[0]
    if len(changes) < 6 or bpm < 96:
        return times
    spacing = float(np.median(np.diff(changes)))
    if spacing >= 7:
        even = np.sum(changes % 2 == 0)
        ph = 0 if even >= len(changes) - even else 1
        log.info("Harmonic rhythm suggests half time (%.0f → %.0f BPM)", bpm, bpm / 2)
        return times[ph::2]
    return times


def _extend(times: np.ndarray, duration: float) -> np.ndarray:
    """Extrapolate the beat grid to cover the whole recording."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return np.arange(0.0, duration + 0.5, 0.5)
    ibi = float(np.median(np.diff(times[:8])))
    head = []
    t = times[0] - ibi
    while t > -0.5 * ibi:
        head.append(t)
        t -= ibi
    ibi_end = float(np.median(np.diff(times[-8:])))
    tail = []
    t = times[-1] + ibi_end
    while t < duration + 2 * ibi_end:
        tail.append(t)
        t += ibi_end
    return np.concatenate([np.array(head[::-1]), times, np.array(tail)])


def _is_compound(mix: np.ndarray, times: np.ndarray) -> bool:
    """True when beats are mostly divided in three (6/8, 12/8 feel)."""
    import librosa

    env = onset_envelope(mix)
    frame_t = librosa.frames_to_time(np.arange(len(env)), sr=SR, hop_length=HOP)
    votes3 = votes2 = 0.0
    background = float(np.median(env)) + 1e-9
    clear3 = 0
    total = 0
    for a, b in zip(times[:-1], times[1:]):
        if b <= a:
            continue
        i0, i1 = np.searchsorted(frame_t, [a, b])
        seg = env[i0:i1]
        if len(seg) < 6:
            continue
        def strength(fr):
            idx = int(fr * len(seg))
            return seg[max(0, idx - 1) : idx + 2].max() if idx < len(seg) else 0.0
        s2 = strength(0.5)
        s3 = (strength(1 / 3) + strength(2 / 3)) / 2
        votes3 += s3
        votes2 += s2
        total += 1
        if s3 > 1.5 * s2 and s3 > 3 * background:
            clear3 += 1
    return bool(total and votes3 > 1.35 * votes2 and clear3 >= 0.45 * total)
