"""Guitar II: bass, harmony and groove.

The accompaniment copies what the band actually plays. Where the piano or
guitar stems strike chords, Guitar II strikes them in the same rhythm; where
they arpeggiate, it arpeggiates the same notes; only where the source holds
long pads (which a guitar cannot sustain) does it fall back to an idiomatic
fingerstyle pattern built from the chord.
"""

from __future__ import annotations

from ..model import TPQ, Bar, Chord, Note, Song
from ..theory import chord_pcs
from .lead import HARMONY_PARTS, energy_at, section_label
from .lines import LineIndex, bassline

COMP_FLOOR = 50  # D3: lowest comp voice
COMP_CEIL = 77  # F5


def lowest_pitch(tuning, capo: int) -> int:
    return min(tuning) + capo


def fit_bass(p: int, lo: int) -> int:
    while p < lo:
        p += 12
    while p > lo + 19:
        p -= 12
    while p - 12 >= lo and p > 55:
        p -= 12
    return p


def bass_line(song: Song, tuning, capo: int) -> list[Note]:
    lo = lowest_pitch(tuning, capo)
    src = bassline(song.bass) if song.bass else []
    out: list[Note] = []
    for n in src:
        out.append(n.copy(pitch=fit_bass(n.pitch, lo), role="bass", priority=20.0, source="bass"))
    covered = LineIndex(out)
    # Chords with no bass note under them get their root (or slash bass).
    for ch in song.chords:
        if ch.end - ch.start < TPQ // 2:
            continue
        if covered.min_between(ch.start, ch.end) is not None:
            continue
        pcb = ch.bass if ch.bass is not None else ch.root
        p = fit_bass(36 + pcb, lo)
        out.append(Note(ch.start, ch.end, p, velocity=0.7, role="bass", priority=16.0, source="chords"))
    out.sort(key=lambda n: n.start)
    # Basses ring until the next bass note (let ring is how fingerstyle bass works)
    for a, b in zip(out, out[1:]):
        if b.start > a.end and b.start - a.end <= TPQ:
            a.end = b.start
    return out


def comp_sources(song: Song, melody_idx: LineIndex) -> list[Note]:
    src = []
    for name, notes in song.parts.items():
        if not any(name.startswith(p) for p in HARMONY_PARTS):
            continue
        for n in notes:
            m = melody_idx.at(n.start)
            if m is not None and (n.pitch - m.pitch) % 12 == 0 and abs(n.start - m.start) <= TPQ // 4:
                continue  # the piano doubling the tune: Guitar I has it
            src.append(n)
    return sorted(src, key=lambda n: (n.start, n.pitch))


def classify_bar(notes: list[Note], bar: Bar) -> str:
    onsets: dict[int, list[Note]] = {}
    for n in notes:
        onsets.setdefault(n.start, []).append(n)
    if len(onsets) <= 2:
        return "sustain"
    multi = sum(1 for v in onsets.values() if len(v) >= 2)
    if multi >= max(2, len(onsets) // 2):
        return "block"
    return "arpeggio"


def place_in_range(p: int, lo: int, hi: int) -> int | None:
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p if lo <= p <= hi else None


def voicing(pcs: list[int], lo: int, hi: int, prev: list[int] | None, voices: int = 3, prefer_top: int | None = None) -> list[int]:
    """Pick ``voices`` chord tones in [lo, hi], spaced, near the previous voicing."""
    cands = sorted(p for p in range(lo, hi + 1) if p % 12 in pcs)
    if not cands:
        return []
    best, best_cost = [], 1e9
    n = len(cands)
    import itertools

    for k in range(min(voices, n), 0, -1):
        for combo in itertools.combinations(cands, k):
            if any(b - a < 3 for a, b in zip(combo, combo[1:])):
                continue
            if combo[-1] - combo[0] > 16:
                continue
            got = {p % 12 for p in combo}
            cost = 2.5 * (len(set(pcs[:3])) - len(got & set(pcs[:3])))
            if pcs and pcs[0] in got:
                cost -= 0.2
            if prev:
                cost += 0.12 * sum(min(abs(p - q) for q in prev) for p in combo)
            else:
                cost += 0.05 * abs(sum(combo) / k - 62)
            if prefer_top is not None:
                cost += 0.15 * abs(combo[-1] - prefer_top)
            if cost < best_cost:
                best, best_cost = list(combo), cost
        if best:
            break
    return best


def chord_segments(song: Song, start: int, end: int) -> list[Chord]:
    return [c for c in song.chords if c.start < end and c.end > start]


def pattern_steps(bar: Bar, tempo: float, percussive: bool, energy: float) -> list[tuple[int, str]]:
    """(offset, voice) pairs for one bar. Voices: B=bass, A=alt bass, 1..3 upper, D=dyad (2+3)."""
    L = bar.length
    if bar.ticks_per_beat != TPQ:  # compound meter: groups of three eighths
        e = TPQ // 2
        per = ["B", "1", "2", "3", "2", "1"]
        return [(i * e, per[i % 6]) for i in range(L // e)]
    if percussive:
        e = TPQ // 2
        seq = []
        for i in range(L // e):
            beat = i // 2
            if i % 2 == 0:
                seq.append((i * e, "B" if beat % 2 == 0 else "A"))
            else:
                seq.append((i * e, "D"))
        return seq
    if tempo < 78 and energy > 0.3:
        s = TPQ // 4
        per = ["B", "1", "2", "3", "1", "2", "3", "2"]
        return [(i * s, per[i % 8] if i % 8 else "B") for i in range(L // s) if i % 2 == 0 or energy > 0.45]
    e = TPQ // 2
    if bar.beats == 3:
        per = ["B", "1", "2", "3", "2", "1"]
    else:
        per = ["B", "1", "2", "3", "A", "1", "2", "3"]
    return [(i * e, per[i % len(per)]) for i in range(L // e)]


def comp_part(song: Song, lead_notes: list[Note], tuning, capo: int) -> list[Note]:
    melody = [n for n in lead_notes if n.role == "melody"]
    mel = LineIndex(melody)
    lead_all = LineIndex(sorted(lead_notes, key=lambda n: n.start))
    bass = bass_line(song, tuning, capo)
    bidx = LineIndex(bass)
    src = comp_sources(song, mel)
    by_bar: dict[int, list[Note]] = {}
    for n in src:
        by_bar.setdefault(song.bar_at(n.start), []).append(n)
    out: list[Note] = list(bass)
    prev_voicing: list[int] | None = None
    for bi, bar in enumerate(song.bars):
        notes = by_bar.get(bi, [])
        texture = classify_bar(notes, bar)
        energy = energy_at(song, bar.start)
        tempo = bar.tempo

        def ceiling(t0: int, t1: int) -> int:
            top = mel.max_between(t0, t1)
            return min(COMP_CEIL, (top - 2) if top is not None else COMP_CEIL)

        def floor(t: int) -> int:
            b = bidx.at(t)
            return max(COMP_FLOOR, (b.pitch + 4) if b is not None else COMP_FLOOR)

        if texture in ("block", "arpeggio"):
            onsets: dict[int, list[Note]] = {}
            for n in notes:
                onsets.setdefault(n.start, []).append(n)
            times = sorted(onsets)
            for k, t in enumerate(times):
                nxt = times[k + 1] if k + 1 < len(times) else bar.end
                group = sorted(onsets[t], key=lambda n: -n.pitch)
                ch = song.chord_at(t)
                lo, hi = floor(t), ceiling(t, max(nxt, t + 1))
                if hi - lo < 3:
                    continue
                if texture == "block":
                    pcs = []
                    for n in group:
                        if n.pitch % 12 not in pcs:
                            pcs.append(n.pitch % 12)
                    if ch is not None:
                        order = chord_pcs(ch.root, ch.quality)
                        pcs = sorted(pcs, key=lambda p: order.index(p) if p in order else 9)
                    v = voicing(pcs, lo, hi, prev_voicing, 3 if energy > 0.35 else 2, prefer_top=place_in_range(group[0].pitch, lo, hi))
                    prev_voicing = v or prev_voicing
                    end = max(nxt, min(t + 2 * bar.ticks_per_beat, max(n.end for n in group)))
                    if ch is not None:
                        end = min(end, ch.end) if ch.end > t else end
                    for j, p in enumerate(v):
                        out.append(Note(t, max(t + TPQ // 4, end), p, velocity=max(n.velocity for n in group) * 0.85, role="comp", priority=3.2 - 0.3 * j, source="comp", tags={"octave_ok"}))
                else:
                    n = group[0]
                    p = place_in_range(n.pitch, lo, hi)
                    if p is None:
                        continue
                    end = min(t + 2 * bar.ticks_per_beat, (ch.end if ch is not None and ch.end > t else bar.end))
                    out.append(Note(t, max(nxt, end), p, velocity=n.velocity * 0.85, role="comp", priority=3.0, source="comp", tags={"octave_ok", "arp"}))
            continue
        # Sustained source (pads/strings) or nothing: idiomatic pattern from the chords.
        steps = pattern_steps(bar, tempo, song.percussive, energy)
        label = section_label(song, bar.start)
        harp = (not song.percussive) and (energy < 0.45 or "intro" in label or "outro" in label)
        for ch in chord_segments(song, bar.start, bar.end):
            pcs = chord_pcs(ch.root, ch.quality)
            seg0, seg1 = max(bar.start, ch.start), min(bar.end, ch.end)
            lo, hi = floor(seg0), ceiling(seg0, seg1)
            if hi - lo < 5:
                continue
            v = voicing(pcs, lo, hi, prev_voicing, 3)
            if not v:
                continue
            prev_voicing = v
            for off, voice in steps:
                t = bar.start + off
                if not seg0 <= t < seg1:
                    continue
                if voice in ("B", "A"):
                    if bidx.at(t) is not None and bidx.at(t).start == t:
                        continue  # the real bass line already speaks here
                    b = bidx.at(t)
                    if b is None:
                        continue
                    if voice == "A" and song.percussive:
                        fifth = (ch.root + 7) % 12
                        p = fit_bass(36 + fifth, lowest_pitch(tuning, capo))
                        if abs(p - b.pitch) > 9:
                            p = b.pitch
                    else:
                        p = b.pitch
                    out.append(Note(t, min(seg1, t + 2 * bar.ticks_per_beat), p, velocity=0.62, role="bass", priority=8.0, source="pattern"))
                    continue
                if voice == "D":
                    pitches = v[-2:]
                else:
                    idx = {"1": 0, "2": 1, "3": 2}[voice]
                    pitches = [v[min(idx, len(v) - 1)]]
                ring_end = seg1 if not song.percussive else min(seg1, t + bar.ticks_per_beat)
                for p in pitches:
                    tags = {"octave_ok", "arp"} | ({"prefer_harmonic"} if harp else set())
                    out.append(Note(t, ring_end, p, velocity=0.55 + 0.2 * energy, role="comp", priority=2.6, source="pattern", tags=tags))
    return resolve_collisions(out, lead_all)


def resolve_collisions(notes: list[Note], lead: LineIndex) -> list[Note]:
    """Drop Guitar II notes that duplicate Guitar I's exact pitch at the same onset."""
    lead_attacks = {(n.start, n.pitch) for n in lead.notes}
    seen = set()
    out = []
    for n in sorted(notes, key=lambda n: (n.start, -n.priority)):
        k = (n.start, n.pitch)
        if k in seen:
            continue
        if n.role != "bass" and k in lead_attacks:
            continue
        seen.add(k)
        out.append(n)
    return out
