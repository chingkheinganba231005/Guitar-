"""Tiny additive synthesiser that renders a Song to audio (for tests/demos)."""

from __future__ import annotations

import numpy as np

from .model import Song

SR = 44100


def _hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _env(n: int, attack: float, release: float, decay: float | None = None) -> np.ndarray:
    t = np.arange(n) / SR
    e = np.minimum(1.0, t / max(attack, 1e-4))
    if decay:
        e *= np.exp(-t / decay)
    r = int(release * SR)
    if r and n > r:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def _tone(pitch, dur, kind, vel):
    n = max(1, int(dur * SR))
    t = np.arange(n) / SR
    f = _hz(pitch)
    if kind == "voice":
        # glottal pulse train through "ah" vowel formants, with vibrato and breath
        vib = 1 + 0.013 * np.sin(2 * np.pi * 5.3 * t) * np.minimum(1, t / 0.3)
        phase = 2 * np.pi * f * np.cumsum(vib) / SR
        sig = np.zeros(n)
        for k in range(1, int(5000 / f)):
            fk = k * f
            amp = sum(a / (1 + ((fk - fc) / bw) ** 2) for fc, bw, a in ((730, 90, 1.0), (1090, 110, 0.5), (2440, 160, 0.25), (3400, 250, 0.08)))
            sig += amp / k**0.3 * np.sin(k * phase)
        sig += 0.02 * np.random.default_rng(int(pitch)).standard_normal(n)
        sig *= _env(n, 0.05, 0.1) * 0.6
    elif kind == "piano":
        sig = sum(np.exp(-t * (1.2 + 0.7 * k)) / k * np.sin(2 * np.pi * f * k * t) for k in range(1, 8))
        sig *= _env(n, 0.004, 0.05)
    elif kind == "bass":
        sig = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
        sig *= _env(n, 0.006, 0.05, 1.4)
    else:  # pad
        sig = sum(np.sin(2 * np.pi * f * k * t + k) / (k * 1.6) for k in range(1, 6))
        sig *= _env(n, 0.25, 0.3)
    return sig * vel


def _drum(kind, vel, rng):
    if kind == "kick":
        n = int(0.28 * SR)
        t = np.arange(n) / SR
        f = 45 + 80 * np.exp(-t * 28)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * vel * 1.4
    if kind == "snare":
        n = int(0.2 * SR)
        t = np.arange(n) / SR
        noise = rng.standard_normal(n)
        noise = np.convolve(noise, [1, -0.8], mode="same")
        return (0.7 * noise * np.exp(-t * 18) + 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25)) * vel
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    noise = np.diff(rng.standard_normal(n + 1))
    return 0.25 * noise * np.exp(-t * 70) * vel


def render(song: Song, path: str | None = None) -> np.ndarray:
    total = song.seconds_at(song.end_tick) + 1.5
    out = np.zeros((2, int(total * SR)))
    rng = np.random.default_rng(7)
    voices = [("vocals", "voice", 0.32, 0.0), ("bass", "bass", 0.45, 0.0), ("piano", "piano", 0.22, -0.3), ("other", "pad", 0.12, 0.3)]
    for part, kind, gain, pan in voices:
        for n in song.parts.get(part, []):
            t0 = song.seconds_at(n.start)
            t1 = song.seconds_at(n.end)
            sig = _tone(n.pitch, max(0.05, t1 - t0) + (0.35 if kind == "piano" else 0.0), kind, gain * (0.5 + n.velocity))
            i = int(t0 * SR)
            j = min(out.shape[1], i + len(sig))
            out[0, i:j] += sig[: j - i] * (1 - pan) * 0.7
            out[1, i:j] += sig[: j - i] * (1 + pan) * 0.7
    for h in song.hits:
        sig = _drum(h.kind, 0.55 * h.velocity, rng)
        i = int(song.seconds_at(h.start) * SR)
        j = min(out.shape[1], i + len(sig))
        out[:, i:j] += sig[: j - i]
    out /= max(1e-9, np.abs(out).max()) / 0.9
    if path:
        import soundfile as sf

        sf.write(path, out.T.astype(np.float32), SR)
    return out
