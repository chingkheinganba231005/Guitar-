"""Line utilities: monophonic reduction, skylines, octave placement."""

from __future__ import annotations

import bisect
from statistics import median

from ..model import Note


def sort_notes(notes):
    return sorted(notes, key=lambda n: (n.start, -n.pitch))


def monophonic(notes: list[Note], prefer: str = "high", min_len: int = 2) -> list[Note]:
    """Reduce to one note at a time. Overlaps are resolved by ``prefer``
    (high, low or loud) at each onset, then by trimming the earlier note."""
    if not notes:
        return []
    key = {
        "high": lambda n: (n.start, -n.pitch),
        "low": lambda n: (n.start, n.pitch),
        "loud": lambda n: (n.start, -n.velocity, -n.pitch),
    }[prefer]
    ordered = sorted(notes, key=key)
    out: list[Note] = []
    for n in ordered:
        if out and out[-1].start == n.start:
            continue  # same onset: first by preference wins
        if out and out[-1].end > n.start:
            prev = out[-1]
            if prefer == "loud" and n.velocity < prev.velocity * 0.6 and n.end <= prev.end:
                continue  # a quiet ghost inside a strong note
            prev.end = n.start
            if prev.end - prev.start < min_len:
                out.pop()
        out.append(n.copy())
    return [n for n in out if n.end - n.start >= min_len]


def skyline(notes: list[Note]) -> list[Note]:
    return monophonic(notes, "high")


def bassline(notes: list[Note]) -> list[Note]:
    return monophonic(notes, "low")


def notes_between(notes: list[Note], start: int, end: int) -> list[Note]:
    return [n for n in notes if n.start < end and n.end > start]


def attacks_between(notes: list[Note], start: int, end: int) -> list[Note]:
    return [n for n in notes if start <= n.start < end]


class LineIndex:
    """Fast "what is sounding at tick t" for a monophonic line."""

    def __init__(self, notes: list[Note]):
        self.notes = sorted(notes, key=lambda n: n.start)
        self.starts = [n.start for n in self.notes]

    def at(self, t: int) -> Note | None:
        i = bisect.bisect_right(self.starts, t) - 1
        if i >= 0 and self.notes[i].end > t:
            return self.notes[i]
        return None

    def next_after(self, t: int) -> Note | None:
        i = bisect.bisect_right(self.starts, t)
        return self.notes[i] if i < len(self.notes) else None

    def prev_before(self, t: int) -> Note | None:
        i = bisect.bisect_left(self.starts, t) - 1
        return self.notes[i] if i >= 0 else None

    def max_between(self, start: int, end: int) -> int | None:
        i = max(0, bisect.bisect_right(self.starts, start) - 1)
        best = None
        while i < len(self.notes) and self.notes[i].start < end:
            n = self.notes[i]
            if n.end > start:
                best = n.pitch if best is None else max(best, n.pitch)
            i += 1
        return best

    def min_between(self, start: int, end: int) -> int | None:
        i = max(0, bisect.bisect_right(self.starts, start) - 1)
        best = None
        while i < len(self.notes) and self.notes[i].start < end:
            n = self.notes[i]
            if n.end > start:
                best = n.pitch if best is None else min(best, n.pitch)
            i += 1
        return best


class PolyIndex:
    """Which pitches sound around a tick, for a polyphonic part."""

    def __init__(self, notes: list[Note]):
        self.notes = sorted(notes, key=lambda n: n.start)
        self.starts = [n.start for n in self.notes]
        self.longest = max((n.end - n.start for n in self.notes), default=0)

    def pitches(self, t0: int, t1: int | None = None) -> set[int]:
        """Pitches attacked or still ringing anywhere in [t0, t1) (default: at t0)."""
        t1 = t0 + 1 if t1 is None else t1
        i = bisect.bisect_left(self.starts, t0 - self.longest)
        j = bisect.bisect_left(self.starts, t1)
        return {n.pitch for n in self.notes[i:j] if n.end > t0}


def best_octave_shift(pitches: list[int], lo: int, hi: int, target: float) -> int:
    """Octave shift that keeps most pitches in [lo, hi] near ``target``."""
    if not pitches:
        return 0
    med = median(pitches)
    best, best_score = 0, -1e9
    for shift in (-36, -24, -12, 0, 12, 24, 36):
        inside = sum(1 for p in pitches if lo <= p + shift <= hi)
        score = inside / len(pitches) - 0.012 * abs(med + shift - target) - (0.02 if shift else 0.0)
        if score > best_score:
            best, best_score = shift, score
    return best


def gaps(notes: list[Note], start: int, end: int, min_gap: int) -> list[tuple[int, int]]:
    """Silent stretches of at least ``min_gap`` ticks inside [start, end)."""
    out = []
    cur = start
    for n in sorted(notes, key=lambda n: n.start):
        if n.end <= start or n.start >= end:
            continue
        if n.start - cur >= min_gap:
            out.append((cur, n.start))
        cur = max(cur, n.end)
    if end - cur >= min_gap:
        out.append((cur, end))
    return out
