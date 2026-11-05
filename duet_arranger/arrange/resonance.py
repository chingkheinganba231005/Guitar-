"""Open-string resonance: the reason open and custom tunings sound huge.

At chord attacks, any open string that is a chord tone, sits below the
melody, does not undercut the bass, does not rub against what is sounding
and is not trapped under a barre gets added — softly — so the chord rings on
every free string. The count of such strings also feeds the tuning search,
so tunings whose open strings belong to the song win when they help.
"""

from __future__ import annotations

from ..model import Song
from ..theory import chord_pcs
from .lines import LineIndex
from .techniques import PlacedNote


def resonance_additions(
    placed: list[PlacedNote],
    song: Song,
    tuning,
    capo: int,
    melody: LineIndex | None,
    max_strings: int = 6,
    others: list | None = None,
) -> list[PlacedNote]:
    """``others`` are the other guitar's notes: an open string never doubles
    one of its attacks nor clashes a semitone or a tone against what it holds."""
    other_attacks = {(n.start, n.pitch) for n in (others or [])}
    other_sorted = sorted(others or [], key=lambda n: n.start)
    by_tick: dict[int, list[PlacedNote]] = {}
    for n in placed:
        by_tick.setdefault(n.start, []).append(n)
    chord_starts = {c.start for c in song.chords}
    out: list[PlacedNote] = []
    ordered = sorted(placed, key=lambda n: n.start)
    for t, attacks in sorted(by_tick.items()):
        pitched = [n for n in attacks if n.fret != "x" and not n.harmonic and "RH" not in n.techs and "LH" not in n.techs]
        if t not in chord_starts and len(pitched) < 2:
            continue
        if any(n.fret == "x" or "RH" in n.techs or n.harmonic == "artificial" for n in attacks):
            continue  # a slap mutes the strings; a tapping or harp-harmonic hand is busy
        ch = song.chord_at(t)
        if ch is None:
            continue
        pcs = set(chord_pcs(ch.root, ch.quality))
        sounding = [n for n in ordered if n.start <= t < n.end and n.fret != "x"]
        if not sounding:
            continue
        used = {n.string for n in sounding}
        pitches = [n.pitch for n in sounding if n.pitch is not None]
        if not pitches:
            continue
        other_here = [n.pitch for n in other_sorted if n.start <= t < n.end] if other_sorted else []
        lowest = min(pitches)
        mel = melody.at(t) if melody else None
        ceiling = (mel.pitch - 1) if mel is not None else 77
        fretted = [n for n in sounding if isinstance(n.fret, (int, float)) and n.fret > 0 and not n.harmonic]
        min_fret = min((n.fret for n in fretted), default=None)
        barre_strings = sorted(n.string for n in fretted if n.fret == min_fret)
        attacked_strings = {n.string for n in attacks if n.fret != "x"}  # harmonics are plucked too
        added = []
        for s in range(1, 7):
            if s in used:
                continue
            op = tuning[s - 1] + capo
            if op % 12 not in pcs:
                continue
            if op < lowest and op % 12 != (ch.bass if ch.bass is not None else ch.root):
                continue
            if op > ceiling:
                continue
            if mel is not None and (mel.pitch - op) % 12 in (1, 11, 2, 10) and mel.pitch - op < 24:
                continue
            if any(abs(op - p) in (1, 2) for p in pitches + other_here):
                continue
            if (t, op) in other_attacks:
                continue
            if len(barre_strings) >= 2 and barre_strings[0] < s < barre_strings[-1]:
                continue
            added.append(s)
        if not added:
            continue
        total = len(attacked_strings) + len(added)
        if total > max_strings:
            added = added[: max(0, max_strings - len(attacked_strings))]
        if not added:
            continue
        strum = total > 4
        end = min(ch.end, t + song.bars[song.bar_at(t)].length)
        for s in added:
            op = tuning[s - 1] + capo
            out.append(PlacedNote(t, end, s, 0, op, "resonance", 0.42, strum=strum, tags={"resonance"}))
        if strum:
            for n in attacks:
                n.strum = True
    return out
