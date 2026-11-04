"""Kick / snare / hat detection on the drum stem and drum prominence."""

from __future__ import annotations

import numpy as np

from .rhythm import HOP, SR, onset_envelope


def _peaks(env: np.ndarray, delta: float, wait: int = 4) -> np.ndarray:
    import librosa

    if env.max() <= 0:
        return np.array([], dtype=int)
    e = env / env.max()
    return librosa.util.peak_pick(e, pre_max=3, post_max=3, pre_avg=12, post_avg=12, delta=delta, wait=wait)


def detect_hits(drums: np.ndarray) -> list[tuple[float, str, float]]:
    """(time, kind, strength 0..1) for kick, snare and hat onsets.

    Onsets come from broadband spectral flux; each is classified by *linear*
    band energies just after it (log flux makes kicks and snares look alike).
    """
    import librosa

    env = onset_envelope(drums)
    peaks = _peaks(env, 0.06, wait=3)
    if not len(peaks):
        return []
    S = np.abs(librosa.stft(drums, n_fft=2048, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    bands = {
        "low": S[(freqs >= 30) & (freqs <= 120)].sum(axis=0),
        "body": S[(freqs > 150) & (freqs <= 1500)].sum(axis=0),
        "noise": S[(freqs > 2000) & (freqs <= 8000)].sum(axis=0),
        "air": S[freqs > 8000].sum(axis=0),
    }
    w = max(1, int(0.05 * SR / HOP))
    rows = []
    for f in peaks:
        e = {k: float(v[f : f + w].max()) if f < len(v) else 0.0 for k, v in bands.items()}
        rows.append((f, e))
    lows = np.array([e["low"] for _, e in rows])
    snr = np.array([e["body"] + e["noise"] for _, e in rows])
    airs = np.array([e["air"] for _, e in rows])
    low_ref = np.percentile(lows, 90) + 1e-12
    sn_ref = np.percentile(snr, 90) + 1e-12
    air_ref = np.percentile(airs, 90) + 1e-12
    out = []
    for (f, e), lo, sn, ai in zip(rows, lows, snr, airs):
        t = float(librosa.frames_to_time(f, sr=SR, hop_length=HOP))
        lo_n, sn_n, ai_n = lo / low_ref, sn / sn_ref, ai / air_ref
        if lo_n > 0.35 and lo > 0.6 * (e["body"] + 0.5 * e["noise"]):
            out.append((t, "kick", float(min(1.0, lo_n))))
            if sn_n > 0.6 and e["noise"] > 0.4 * lo:
                out.append((t, "snare", float(min(1.0, sn_n))))
        elif sn_n > 0.35 or (sn_n > 0.2 and (e["body"] + e["noise"]) > 2.0 * e["air"]):
            out.append((t, "snare", float(min(1.0, sn_n))))
        elif ai_n > 0.15:
            out.append((t, "hat", float(min(1.0, ai_n))))
    return sorted(out)


def prominence(mix: np.ndarray, drums: np.ndarray) -> float:
    """0..1: how much of the song's energy and pulse comes from the drums."""
    rms_mix = float(np.sqrt(np.mean(mix**2)) + 1e-9)
    rms_d = float(np.sqrt(np.mean(drums**2)))
    ratio = rms_d / rms_mix
    return float(np.clip(ratio / 0.45, 0.0, 1.0))
