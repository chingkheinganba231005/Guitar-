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
    beats: np.ndarray  # seconds of every tracked pulse
    beats_per_bar: int  # pulses per bar
    downbeat_phase: int
    compound: bool  # pulses are dotted quarters (6/8, 12/8 felt in two/four)
    eighths: str = ""  # "6/8" or "3/4" when the pulses are eighth notes, six to a bar

    @property
    def ticks_per_beat(self) -> int:
        """Ticks of the felt beat of the written meter."""
        if self.eighths == "6/8":
            return int(TPQ * 1.5)
        if self.eighths == "3/4":
            return TPQ
        return int(TPQ * 1.5) if self.compound else TPQ

    @property
    def pulse_ticks(self) -> int:
        """Ticks between two tracked pulses."""
        return TPQ // 2 if self.eighths else self.ticks_per_beat

    def bars(self) -> list[Bar]:
        """Bars on the tick grid; beat 0 is tick 0, a pickup bar covers beats before the first downbeat."""
        tpb = self.ticks_per_beat
        pt = self.pulse_ticks
        b = self.beats
        m = self.beats_per_bar
        out: list[Bar] = []
        per_quarter = 0.5 if self.eighths else (1.5 if self.compound else 1.0)

        def tempo_of(i0: int, i1: int) -> float:
            seg = b[i0 : min(i1 + 1, len(b))]
            ibi = float(np.median(np.diff(seg))) if len(seg) >= 2 else float(np.median(np.diff(b)))
            return float(np.clip(60.0 / ibi * per_quarter, 20, 300))

        ph = self.downbeat_phase
        if self.eighths:
            beats, unit = (6, 8) if self.eighths == "6/8" else (3, 4)
            if ph > 0:
                out.append(Bar(0, ph, 8, tempo_of(0, ph), TPQ // 2))
            for k in range(ph, len(b) - 1, m):
                out.append(Bar(k * pt, beats, unit, tempo_of(k, k + m), tpb))
            smooth_tempos(out)
            return out
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
        smooth_tempos(out)
        return out

    def to_ticks(self, t: float) -> float:
        """Map seconds to ticks by linear interpolation between beats (beat i = tick i*tpb)."""
        b = self.beats
        tpb = self.pulse_ticks
        if t <= b[0]:
            return 0.0
        if t >= b[-1]:
            ibi = b[-1] - b[-2]
            return (len(b) - 1 + (t - b[-1]) / ibi) * tpb
        j = int(np.searchsorted(b, t) - 1)
        frac = (t - b[j]) / (b[j + 1] - b[j])
        return (j + frac) * tpb


def smooth_tempos(bars: list[Bar], window: int = 9, steady: float = 0.05) -> None:
    """One tempo for the song unless it really changes.

    Per-bar tempos from a beat tracker wobble by a few percent; written into
    the score they make every bar a tempo change. Notes are already placed
    against the real beats, so the tempo only sets playback speed: use the
    song's median unless a rolling median drifts more than ``steady`` away
    for a whole window, and then change in whole BPM steps only.
    """
    if not bars:
        return
    raw = np.array([b.tempo for b in bars], dtype=float)
    glob = float(np.median(raw))
    half = window // 2
    rolling = np.array([np.median(raw[max(0, i - half) : i + half + 1]) for i in range(len(raw))])
    cur = round(glob)
    for b, r in zip(bars, rolling):
        if abs(r - glob) / glob > steady:
            if abs(r - cur) / cur > 0.03:
                cur = round(r)
        elif abs(cur - glob) / glob > steady:
            cur = round(glob)
        b.tempo = float(cur)


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
    times0 = fix_tempo_octave(np.asarray(times, dtype=float), low_env, env=onset_envelope(mix, detrend=False))
    duration = len(mix) / SR
    first_onset = None
    if len(env):
        idx = np.nonzero(env > 0.3 * env.max())[0]
        if len(idx):
            first_onset = float(librosa.frames_to_time(idx[0], sr=SR, hop_length=HOP))
    sources = (env, drums, bass, harmonic, notes)
    cand = _extend(times0, duration)
    best = choose_meter(mix, cand, sources, first_onset)
    if not best.eighths and notes:
        halved = harmonic_rhythm_octave(times0, notes)
        if len(halved) != len(times0):
            cand = _extend(halved, duration)
            best = choose_meter(mix, cand, sources, first_onset)
    log.info(
        "Beats: %d, %s, %d per bar, phase %d",
        len(best.beats),
        f"eighth pulses in {best.eighths}" if best.eighths else ("compound" if best.compound else "simple"),
        best.beats_per_bar,
        best.downbeat_phase,
    )
    return best


def accent_features(times: np.ndarray, sources) -> tuple[np.ndarray, np.ndarray]:
    """Per-pulse downbeat evidence (kick, bass onsets, chord changes) and plain onset strength."""
    import librosa

    env, drums, bass, harmonic, notes = sources
    frames = np.clip(librosa.time_to_frames(times, sr=SR, hop_length=HOP), 0, len(env) - 1)

    def at_beats(e):
        if e is None:
            return np.zeros(len(times))
        out = np.zeros(len(times))
        for k, f in enumerate(frames):
            out[k] = e[max(0, f - 2) : f + 3].max() if len(e) else 0.0
        sd = out.std()
        return (out - out.mean()) / sd if sd > 0 else out * 0

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
    onset = at_beats(onset_envelope(harmonic, detrend=False) if harmonic is not None else env) + 0.5 * bass_acc
    return accent, onset


def choose_meter(mix: np.ndarray, times: np.ndarray, sources, first_onset: float | None) -> BeatMap:
    """Pick pulses per bar and the downbeat.

    Tried: 2, 3 or 4 pulses per bar, and — when the pulses are fast (100+
    per minute) — six pulses per bar read as eighth notes. Triple-time
    ballads (6/8 at a dotted quarter of ~50) are tracked at the eighth note
    (~150); grouping those in threes makes every bar half as long as written
    and pairs the harmony with the wrong beats. Chord and bass changes every
    six pulses make the six-pulse reading win; where the secondary accent
    then falls (pulse 4: 6/8, pulses 3 and 5: 3/4) names the meter.
    """
    accent, onset = accent_features(times, sources)
    first_beat = int(np.argmin(np.abs(times - first_onset))) if first_onset is not None else None
    bpm = 60.0 / float(np.median(np.diff(times))) if len(times) > 2 else 90.0
    compound = _is_compound(mix, times)
    per_size: dict[int, tuple] = {}
    sizes = [4, 3, 2] + ([6] if bpm >= 100 and not compound else [])
    for m in sizes:
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
                score += 0.45  # songs start on beat 1 far more often than with a pickup
            if m not in per_size or score > per_size[m][0]:
                per_size[m] = (score, m, ph)
    log.info("Meter scores (pulses per bar: score, phase): %s", {k: (round(float(v[0]), 3), v[2]) for k, v in sorted(per_size.items())})
    # Six pulses only when they explain the accents clearly better than three:
    # the harmony moves every six pulses, not every three (a fast 3/4 waltz).
    if 6 in per_size and 3 in per_size:
        s6, s3 = per_size[6][0], per_size[3][0]
        if not (s6 >= 1.2 * s3 and s6 - s3 >= 0.3):
            del per_size[6]
    _, m, ph = max(per_size.values(), key=lambda x: x[0])
    if m == 6:
        pos = (np.arange(len(onset)) - ph) % 6
        mean = lambda ks: float(np.mean([onset[pos == k].mean() for k in ks if np.any(pos == k)])) if any(np.any(pos == k) for k in ks) else 0.0
        six8 = mean([3]) - mean([1, 2, 4, 5])
        three4 = mean([2, 4]) - mean([1, 3, 5])
        return BeatMap(np.asarray(times), 6, ph, False, "6/8" if six8 >= three4 else "3/4")
    if compound and m == 3:
        m = 2
    return BeatMap(np.asarray(times), m, ph, compound)


def fix_tempo_octave(times: np.ndarray, low_env: np.ndarray, lo: float = 62.0, hi: float = 170.0, env: np.ndarray | None = None) -> np.ndarray:
    """Undo double/half-time beat tracking: keep the tempo in [lo, hi] BPM.

    When halving, keep the beat phase where kick/bass onsets land. A slow
    beat is split in three (its eighth notes) when the music divides it in
    three — a 6/8 ballad tracked at the dotted quarter — and in two otherwise."""
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
            if env is not None and _divides_in_three(env, times):
                a, b = times[:-1], times[1:]
                times = np.sort(np.concatenate([times, a + (b - a) / 3, a + 2 * (b - a) / 3]))
            else:
                mids = (times[:-1] + times[1:]) / 2
                times = np.sort(np.concatenate([times, mids]))
        else:
            break
    return times


def _divides_in_three(env: np.ndarray, times: np.ndarray) -> bool:
    """Do onsets fall at thirds of the beat more than at its middle?"""
    import librosa

    frame_t = librosa.frames_to_time(np.arange(len(env)), sr=SR, hop_length=HOP)

    def strength(t):
        i = int(np.searchsorted(frame_t, t))
        return float(env[max(0, i - 2) : i + 3].max()) if i < len(env) else 0.0

    s2 = s3 = 0.0
    for a, b in zip(times[:-1], times[1:]):
        s2 += strength(a + (b - a) / 2)
        s3 += (strength(a + (b - a) / 3) + strength(a + 2 * (b - a) / 3)) / 2
    return s3 > 1.2 * s2


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
        log.info("Harmonic rhythm suggests half time (%.0f to %.0f BPM)", bpm, bpm / 2)
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
