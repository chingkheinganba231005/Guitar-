"""Rhythmic grid: per-beat straight or triplet subdivision and snapping."""

from __future__ import annotations

import bisect

from .model import TPQ, Note, Song


class Grid:
    def __init__(self, song: Song):
        self.song = song
        self.beats = song.beat_starts()
        self.lengths = []
        for i, b in enumerate(self.beats):
            nxt = self.beats[i + 1] if i + 1 < len(self.beats) else song.end_tick
            self.lengths.append(max(1, nxt - b))

    def beat_index(self, t: float) -> int:
        i = bisect.bisect_right(self.beats, t) - 1
        return max(0, min(i, len(self.beats) - 1))

    def is_triplet(self, i: int) -> bool:
        return i in self.song.triplet_beats

    def step(self, i: int) -> int:
        L = self.lengths[i]
        if L != TPQ:  # compound beat (dotted quarter) or odd bar remainder
            return 6 if L % 6 == 0 else 3
        return 4 if self.is_triplet(i) else 3

    def snap(self, t: float) -> int:
        if not self.beats:
            return int(round(t / 3) * 3)
        if t >= self.song.end_tick:
            return self.song.end_tick
        i = self.beat_index(t)
        b = self.beats[i]
        st = self.step(i)
        off = round((t - b) / st) * st
        return int(min(b + off, b + self.lengths[i]))


def snap_notes(notes: list[Note], song: Song, min_len: int = 3) -> list[Note]:
    g = Grid(song)
    out = []
    for n in notes:
        s = g.snap(n.start)
        e = g.snap(n.end)
        if s >= song.end_tick:
            continue
        if e <= s:
            e = s + max(min_len, g.step(g.beat_index(s)))
        n.start, n.end = s, min(e, song.end_tick)
        if n.end > n.start:
            out.append(n)
    return out


def detect_triplet_beats(song: Song, onsets: list[float], bias: float = 0.6) -> set[int]:
    """Beats whose onsets fit a triplet subdivision clearly better than sixteenths.

    ``onsets`` are tick positions before snapping. ``bias`` (<1) favours the
    straight grid so that only convincing triplets/swing are notated as such.
    """
    g = Grid(song)
    per_beat: dict[int, list[float]] = {}
    for t in onsets:
        i = g.beat_index(t)
        if g.lengths[i] != TPQ:
            continue
        per_beat.setdefault(i, []).append((t - g.beats[i]) / TPQ)
    out = set()
    swing_votes = 0
    for i, offs in per_beat.items():
        inner = [o for o in offs if 0.08 < o < 0.92]
        if not inner:
            continue
        e16 = sum(min(abs(o - k / 4) for k in range(5)) for o in inner)
        e3 = sum(min(abs(o - k / 3) for k in range(4)) for o in inner)
        if e3 < e16 * bias:
            out.add(i)
            swing_votes += 1
    return out
