"""Core data structures shared by analysis, arrangement and export.

Time is measured in integer *ticks*. ``TPQ`` ticks make one quarter note, which
lets a single grid hold 32nd notes (3 ticks), sixteenths (6), triplet eighths
(8) and triplet sixteenths (4) without rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Optional

TPQ = 24  # ticks per quarter note

_ids = count(1)


def next_id() -> int:
    return next(_ids)


def reserve_ids(above: int) -> None:
    """Make future ids larger than ``above`` (after unpickling a saved Song)."""
    global _ids
    current = next(_ids)
    _ids = count(max(current, above + 1))


@dataclass
class Note:
    """A pitched note in sounding (concert) pitch."""

    start: int
    end: int
    pitch: int
    velocity: float = 0.7  # 0..1
    role: str = ""  # melody, bass, harmony, comp, fill, harmonic, support
    priority: float = 1.0
    source: str = ""  # stem or track it came from
    tags: set = field(default_factory=set)
    id: int = field(default_factory=next_id)

    @property
    def duration(self) -> int:
        return self.end - self.start

    def copy(self, **changes) -> "Note":
        data = dict(
            start=self.start,
            end=self.end,
            pitch=self.pitch,
            velocity=self.velocity,
            role=self.role,
            priority=self.priority,
            source=self.source,
            tags=set(self.tags),
        )
        data.update(changes)
        return Note(**data)


@dataclass
class Hit:
    """A percussion event (kick, snare, hat) on the tick grid."""

    start: int
    kind: str  # kick | snare | hat
    velocity: float = 0.7
    id: int = field(default_factory=next_id)


@dataclass
class Chord:
    start: int
    end: int
    root: int  # pitch class 0..11
    quality: str  # maj, min, 7, maj7, m7, sus2, sus4, dim, aug, 5, m7b5, 6, m6, add9
    bass: Optional[int] = None  # pitch class of the bass if not the root

    @property
    def pcs(self) -> list[int]:
        from .theory import chord_pcs

        return chord_pcs(self.root, self.quality)

    def name(self, prefer_flats: bool = False) -> str:
        from .theory import chord_name

        return chord_name(self.root, self.quality, self.bass, prefer_flats)


@dataclass
class Section:
    start_bar: int
    label: str


@dataclass
class Bar:
    """One measure of the shared bar grid."""

    start: int  # tick
    beats: int  # numerator
    unit: int  # denominator (4 or 8)
    tempo: float  # quarter-note BPM at this bar
    ticks_per_beat: int  # TPQ for simple meters, 1.5*TPQ for compound

    @property
    def length(self) -> int:
        return self.beats * TPQ * 4 // self.unit

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def meter(self) -> tuple[int, int]:
        return (self.beats, self.unit)

    @property
    def pulses(self) -> int:
        """Number of felt beats (quarter notes, or dotted quarters in compound meter)."""
        return max(1, self.length // self.ticks_per_beat)


@dataclass
class Song:
    """Everything known about the source recording, on a shared tick grid."""

    title: str = "Untitled"
    artist: str = ""
    bars: list[Bar] = field(default_factory=list)
    key_tonic: int = 0  # pitch class
    key_mode: str = "major"  # major | minor
    parts: dict[str, list[Note]] = field(default_factory=dict)  # stem -> notes
    melody: list[Note] = field(default_factory=list)
    bass: list[Note] = field(default_factory=list)
    chords: list[Chord] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    percussive: bool = False  # drums are a significant part of the song
    drum_prominence: float = 0.0
    energy: list[float] = field(default_factory=list)  # per bar, 0..1
    triplet_beats: set = field(default_factory=set)  # global beat indices with triplet grid
    notes_info: dict = field(default_factory=dict)  # free-form analysis report

    @property
    def end_tick(self) -> int:
        return self.bars[-1].end if self.bars else 0

    def bar_at(self, tick: int) -> int:
        lo, hi = 0, len(self.bars) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.bars[mid].start <= tick:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def beat_starts(self) -> list[int]:
        out = []
        for b in self.bars:
            for k in range(b.pulses):
                out.append(b.start + k * b.ticks_per_beat)
        return out

    def seconds_at(self, tick: float) -> float:
        """Convert a tick to seconds using the per-bar tempo map."""
        if not self.bars:
            return tick / TPQ * 0.5
        sec = 0.0
        for b in self.bars:
            spt = 60.0 / b.tempo / TPQ
            if tick <= b.end:
                return sec + max(0.0, tick - b.start) * spt
            sec += b.length * spt
        last = self.bars[-1]
        return sec + (tick - last.end) * 60.0 / last.tempo / TPQ

    def chord_at(self, tick: int) -> Optional[Chord]:
        for c in self.chords:
            if c.start <= tick < c.end:
                return c
        return None

    def section_at_bar(self, bar: int) -> Optional[Section]:
        cur = None
        for s in self.sections:
            if s.start_bar <= bar:
                cur = s
        return cur


@dataclass
class GuitarPart:
    """The notes one guitar should try to play, before fingering."""

    name: str
    notes: list[Note] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    tuning: tuple = (64, 59, 55, 50, 45, 40)
    capo: int = 0
    pan: float = 0.0
