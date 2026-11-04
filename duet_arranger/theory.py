"""Pitch, key and chord helpers."""

from __future__ import annotations

import numpy as np

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "m7b5": (0, 3, 6, 10),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "add9": (0, 2, 4, 7),
    "5": (0, 7),
}

CHORD_SUFFIX = {
    "maj": "",
    "min": "m",
    "7": "7",
    "maj7": "maj7",
    "m7": "m7",
    "sus2": "sus2",
    "sus4": "sus4",
    "dim": "dim",
    "aug": "aug",
    "m7b5": "m7b5",
    "6": "6",
    "m6": "m6",
    "add9": "add9",
    "5": "5",
}

# Krumhansl–Kessler key profiles.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

FLAT_KEYS_MAJOR = {5, 10, 3, 8, 1, 6}  # F Bb Eb Ab Db Gb


def pc(p: int) -> int:
    return p % 12


def note_name(midi: int, prefer_flats: bool = False) -> str:
    names = FLAT_NAMES if prefer_flats else SHARP_NAMES
    return f"{names[midi % 12]}{midi // 12 - 1}"


def pc_name(p: int, prefer_flats: bool = False) -> str:
    return (FLAT_NAMES if prefer_flats else SHARP_NAMES)[p % 12]


def chord_pcs(root: int, quality: str) -> list[int]:
    return [(root + i) % 12 for i in CHORD_INTERVALS.get(quality, (0, 4, 7))]


def chord_name(root: int, quality: str, bass: int | None = None, prefer_flats: bool = False) -> str:
    s = pc_name(root, prefer_flats) + CHORD_SUFFIX.get(quality, "")
    if bass is not None and bass % 12 != root % 12:
        s += "/" + pc_name(bass, prefer_flats)
    return s


def prefers_flats(tonic: int, mode: str) -> bool:
    rel = tonic if mode == "major" else (tonic + 3) % 12
    return rel in FLAT_KEYS_MAJOR


def detect_key(pc_weights: np.ndarray) -> tuple[int, str, float]:
    """Return (tonic, mode, confidence) from a 12-bin pitch-class histogram."""
    w = np.asarray(pc_weights, dtype=float)
    if w.sum() <= 0:
        return 0, "major", 0.0
    best = (-2.0, 0, "major")
    scores = []
    for tonic in range(12):
        for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = float(np.corrcoef(np.roll(prof, tonic), w)[0, 1])
            scores.append(r)
            if r > best[0]:
                best = (r, tonic, mode)
    scores.sort(reverse=True)
    conf = scores[0] - scores[1] if len(scores) > 1 else 0.0
    return best[1], best[2], conf


def scale_pcs(tonic: int, mode: str) -> list[int]:
    steps = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    return [(tonic + s) % 12 for s in steps]


def chord_templates(qualities: tuple[str, ...] = ("maj", "min", "7", "maj7", "m7", "sus2", "sus4", "dim")):
    """Binary-ish templates (root weighted) for every root and quality."""
    labels = []
    mats = []
    for q in qualities:
        for r in range(12):
            t = np.zeros(12)
            for k, i in enumerate(CHORD_INTERVALS[q]):
                t[(r + i) % 12] = 1.0 if k else 1.25
            t /= np.linalg.norm(t)
            labels.append((r, q))
            mats.append(t)
    return labels, np.array(mats)


# Simpler chords are more likely a priori; this keeps the recogniser from
# over-explaining passing tones as extensions.
QUALITY_PRIOR = {"maj": 0.0, "min": 0.0, "7": -0.05, "m7": -0.05, "maj7": -0.06, "sus2": -0.08, "sus4": -0.08, "dim": -0.12}


def recognise_chords(
    chroma: np.ndarray,
    key: tuple[int, str] | None = None,
    bass_chroma: np.ndarray | None = None,
    change_penalty: float = 0.18,
) -> list[tuple[int, str, int | None]]:
    """Viterbi chord labelling over frames of a (n, 12) chroma matrix.

    Returns one (root, quality, bass_pc) per frame. ``bass_chroma`` (n, 12)
    biases roots toward the sounding bass and yields slash chords.
    """
    labels, T = chord_templates()
    n = chroma.shape[0]
    if n == 0:
        return []
    X = chroma / (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9)
    emit = X @ T.T  # (n, L)
    prior = np.array([QUALITY_PRIOR.get(q, -0.1) for _, q in labels])
    emit = emit + prior
    if key is not None:
        sc = set(scale_pcs(*key))
        diat = np.array([0.04 if all(p in sc for p in chord_pcs(r, q)) else 0.0 for r, q in labels])
        emit = emit + diat
    if bass_chroma is not None:
        B = bass_chroma / (bass_chroma.sum(axis=1, keepdims=True) + 1e-9)
        roots = np.array([r for r, _ in labels])
        emit = emit + 0.15 * B[:, roots]
    L = len(labels)
    score = emit[0].copy()
    back = np.zeros((n, L), dtype=int)
    for t in range(1, n):
        best_prev = int(np.argmax(score))
        stay = score
        switch = score[best_prev] - change_penalty
        choose_switch = switch > stay
        back[t] = np.where(choose_switch, best_prev, np.arange(L))
        score = np.where(choose_switch, switch, stay) + emit[t]
    path = [int(np.argmax(score))]
    for t in range(n - 1, 0, -1):
        path.append(back[t][path[-1]])
    path.reverse()
    out = []
    for t, li in enumerate(path):
        r, q = labels[li]
        bass = None
        if bass_chroma is not None and bass_chroma[t].sum() > 0:
            b = int(np.argmax(bass_chroma[t]))
            if b != r and b in chord_pcs(r, q):
                bass = b
        out.append((r, q, bass))
    return out


def nearest_pitch(pc_target: int, around: int) -> int:
    """MIDI pitch with pitch class ``pc_target`` closest to ``around``."""
    base = around - ((around - pc_target) % 12)
    return base if around - base <= 6 else base + 12


def interval_class(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)
