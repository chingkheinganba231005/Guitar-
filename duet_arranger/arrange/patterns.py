"""Regular rhythm patterns distilled from transcribed (noisy) parts.

Transcription gives every onset with its timing wobble and occasional
spurious notes. Copied one-to-one, that becomes rhythms nobody can play. A
band, though, repeats its patterns: for every section we count how often
each eighth-note slot of the bar is struck and keep the slots struck in most
bars. The result is the section's groove — a piano striking each beat, an
arpeggio in eighths, a syncopated stab — written the same way in every bar.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import TPQ, Note, Song


@dataclass
class Pattern:
    slots: list[int]  # slot indices inside the bar
    unit: int  # ticks per slot
    voices: float  # average number of different pitches struck per used slot
    coverage: float  # share of the section's bars where the part plays at all
    length: int = 0  # bar length (ticks) the pattern belongs to
    slot_voices: tuple = ()  # average pitches struck at each pattern slot

    @property
    def block(self) -> bool:
        return self.voices >= 1.8

    def ticks(self) -> list[int]:
        return [s * self.unit for s in self.slots]


def section_ranges(song: Song) -> list[tuple[int, int]]:
    """(first bar, last bar + 1) for every section (the whole song if none)."""
    if not song.sections:
        return [(0, len(song.bars))]
    starts = sorted({max(0, s.start_bar) for s in song.sections} | {0})
    ends = starts[1:] + [len(song.bars)]
    return [(a, b) for a, b in zip(starts, ends) if b > a]


def slot_unit(song: Song, bar_index: int) -> int:
    bar = song.bars[bar_index]
    if bar.ticks_per_beat != TPQ:
        return 12  # eighths in compound meters
    first = song.beat_starts().index(bar.start) if bar.start in song.beat_starts() else None
    if first is not None:
        trip = sum(1 for k in range(bar.pulses) if (first + k) in song.triplet_beats)
        if trip > bar.pulses / 2:
            return 8  # triplet eighths
    return 12


def find_pattern(song: Song, notes: list[Note], bars: range, min_share: float = 0.5) -> Pattern | None:
    """The recurring onset slots of ``notes`` over ``bars`` (bars of one length only)."""
    if not len(bars):
        return None
    # use the most common bar length in the range (skips pickup bars)
    lengths: dict[int, int] = {}
    for bi in bars:
        lengths[song.bars[bi].length] = lengths.get(song.bars[bi].length, 0) + 1
    L = max(lengths, key=lengths.get)
    use = [bi for bi in bars if song.bars[bi].length == L]
    unit = slot_unit(song, use[0])
    nslots = max(1, L // unit)
    by_bar: dict[int, dict[int, set]] = {bi: {} for bi in use}
    for n in notes:
        if n.end - n.start <= 6 and n.velocity < 0.7:
            continue  # a short, quiet blip: transcription debris, not the groove
        bi = song.bar_at(n.start)
        if bi not in by_bar:
            continue
        off = n.start - song.bars[bi].start
        s = int(round(off / unit))
        if s >= nslots:
            continue
        by_bar[bi].setdefault(s, set()).add(n.pitch)
    active = [bi for bi in use if by_bar[bi]]
    if not active:
        return None
    counts = [0] * nslots
    voices = [0] * nslots
    for bi in active:
        for s, ps in by_bar[bi].items():
            counts[s] += 1
            voices[s] += len(ps)
    share = [c / len(active) for c in counts]
    if len(active) < 8:
        min_share = max(min_share, 0.6)  # few bars: a stray note must not become the groove
    slots = [s for s in range(nslots) if share[s] >= min_share]
    if 0 not in slots and share[0] >= 0.3:
        slots.insert(0, 0)
    if not slots:
        # at least the strongest slot, if it is common enough
        best = max(range(nslots), key=lambda s: share[s])
        if share[best] < 0.3:
            return None
        slots = [best]
    used_counts = sum(counts[s] for s in slots) or 1
    avg_voices = sum(voices[s] for s in slots) / used_counts
    per_slot = tuple(voices[s] / counts[s] for s in sorted(slots) if counts[s])
    return Pattern(sorted(slots), unit, avg_voices, len(active) / len(use), L, per_slot)


def chordal(song: Song, notes: list[Note], bars: range) -> bool:
    """True when the part strikes chords rather than playing a line.

    Nearly every slot of the groove must be a chord: a tune over held chords
    (single notes on most slots) is a line, and Guitar I plays it as one."""
    p = find_pattern(song, notes, bars)
    if p is None or p.coverage < 0.5:
        return False
    clusters: dict[int, set] = {}
    for n in notes:
        clusters.setdefault(n.start, set()).add(n.pitch)
    singles = sum(1 for ps in clusters.values() if len(ps) == 1)
    if clusters and singles > 0.35 * len(clusters):
        return False  # a tune is moving between the chords
    return bool(p.slot_voices) and sum(1 for v in p.slot_voices if v >= 1.8) >= 0.75 * len(p.slot_voices)
