"""Guitar II: harmony, groove and the bass in between.

The accompaniment follows what the band plays, but as a groove a guitarist
can repeat: for every section the recurring onset pattern of the piano and
guitar stems is found (see ``patterns``). Where they strike chords, Guitar II
strikes chords on that pattern; where they play single notes, it plays an
arpeggio on that pattern; where they hold pads (which a guitar cannot
sustain) it falls back to an idiomatic fingerstyle pattern. Where Guitar I
carries the chords itself (an instrumental intro, say), Guitar II keeps to
the bass so the two parts do not pile up. Guitar I is a complete
fingerstyle part with its own bass, so Guitar II never doubles a note Guitar I
actually plays: it takes the bass groove between Guitar I's bass notes and
fills the beats Guitar I's bass covers with its pattern's chord tones.
"""

from __future__ import annotations

from ..model import TPQ, Bar, Chord, Note, Song
from ..theory import chord_pcs
from .lead import HARMONY_PARTS, energy_at, section_label
from .lines import LineIndex, PolyIndex, bassline
from .patterns import find_pattern, section_ranges

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
    """The bass on the section's recurring bass rhythm (plus every chord change)."""
    lo = lowest_pitch(tuning, capo)
    src = bassline(song.bass) if song.bass else []
    out: list[Note] = []
    chord_starts = sorted(c.start for c in song.chords)
    for a, b in section_ranges(song):
        sec = [n for n in src if a <= song.bar_at(n.start) < b]
        pat = find_pattern(song, sec, range(a, b), min_share=0.4)
        idx = LineIndex(sec)
        for bi in range(a, b):
            bar = song.bars[bi]
            ticks = {bar.start + t for t in (pat.ticks() if pat else [0]) if t < bar.length}
            ticks |= {t for t in chord_starts if bar.start <= t < bar.end}
            for t in sorted(ticks):
                n = idx.at(t)
                if n is None or n.start < t - TPQ:
                    nxt = idx.next_after(t - 1)
                    n = nxt if nxt is not None and nxt.start - t <= TPQ // 4 else n
                ch = song.chord_at(t)
                if n is not None:
                    pitch = n.pitch
                elif ch is not None:
                    pitch = 36 + (ch.bass if ch.bass is not None else ch.root)
                else:
                    continue
                if ch is not None and n is not None and pitch % 12 not in ch.pcs and (ch.start == t):
                    pitch = 36 + (ch.bass if ch.bass is not None else ch.root)
                out.append(Note(t, t + TPQ, fit_bass(pitch, lo), velocity=(n.velocity if n else 0.7), role="bass", priority=20.0, source="bass"))
    out.sort(key=lambda n: n.start)
    dedup: list[Note] = []
    for n in out:
        if dedup and dedup[-1].start == n.start:
            continue
        dedup.append(n)
    # A bass rings until the next bass note (let ring is how fingerstyle bass works).
    for x, y in zip(dedup, dedup[1:]):
        x.end = max(x.start + 6, y.start)
    if dedup:
        dedup[-1].end = max(dedup[-1].end, min(song.end_tick, dedup[-1].start + song.bars[-1].length))
    return dedup


def comp_sources(song: Song, melody_idx: LineIndex) -> list[Note]:
    from .lead import main_harmony_parts

    src = []
    keep = set(main_harmony_parts(song))
    for name, notes in song.parts.items():
        if name not in keep:
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


def voicing(pcs: list[int], lo: int, hi: int, prev: list[int] | None, voices: int = 3, prefer_top: int | None = None, avoid: set | None = None) -> list[int]:
    """Pick ``voices`` chord tones in [lo, hi], spaced, near the previous voicing
    (never a pitch in ``avoid``: what the other guitar is already sounding)."""
    cands = sorted(p for p in range(lo, hi + 1) if p % 12 in pcs and not (avoid and p in avoid))
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


ARP_ORDER = [0, 1, 2, 1]


def harp_voicing(pcs: list[int], tuning, capo: int, lo: int, hi: int, prev: list[int] | None, avoid: set | None = None) -> list[int]:
    """Chord tones that sit on the tuning's 12th/7th-fret harmonics (bell-like arpeggios)."""
    from .lead import natural_harmonic_pitches

    nh = sorted(p for p in natural_harmonic_pitches(tuning, capo, (12, 7)) if p % 12 in pcs and lo <= p <= hi + 12 and not (avoid and p in avoid))
    if len(nh) < 2:
        return []
    best, cost = [], 1e9
    import itertools

    for k in (3, 2):
        for combo in itertools.combinations(nh, k):
            if combo[-1] - combo[0] > 19 or any(b - a < 3 for a, b in zip(combo, combo[1:])):
                continue
            c = (3 - len({p % 12 for p in combo})) * 2.0
            if prev:
                c += 0.08 * sum(min(abs(p - q) for q in prev) for p in combo)
            if c < cost:
                best, cost = list(combo), c
        if best:
            break
    return best


def comp_part(song: Song, lead_notes: list[Note], tuning, capo: int) -> list[Note]:
    melody = [n for n in lead_notes if n.role == "melody"]
    mel = LineIndex(melody)
    lead_all = LineIndex(sorted(lead_notes, key=lambda n: n.start))
    lead_chord_bars = {song.bar_at(n.start) for n in lead_notes if n.role == "chord"}
    # Guitar I plays its own bass on the strong beats; where it sounds the very
    # same bass note Guitar II leaves it to Guitar I and keeps the bass groove
    # in between (and its pattern takes that slot with a chord tone instead).
    lead_attacks = {(n.start, n.pitch) for n in lead_notes}
    lead_bass = LineIndex([n for n in lead_notes if n.role == "bass"])
    lead_poly = PolyIndex(lead_notes)
    # (Where Guitar I strikes the band's chords instead, Guitar II keeps the
    # whole bass line: those bars are the chords plus Guitar II's bass and bells.)
    bass = [n for n in bass_line(song, tuning, capo) if (n.start, n.pitch) not in lead_attacks or song.bar_at(n.start) in lead_chord_bars]
    bidx = LineIndex(bass)
    src = comp_sources(song, mel)
    out: list[Note] = list(bass)
    prev_voicing: list[int] | None = None

    def ceiling(t0: int, t1: int) -> int:
        top = mel.max_between(t0, t1)
        return min(COMP_CEIL, (top - 2) if top is not None else COMP_CEIL)

    lead_inner = PolyIndex([n for n in lead_notes if n.role == "inner"])

    def floor(t: int, t1: int | None = None, pcs: list[int] | None = None) -> int:
        lows = [b.pitch + 4 for b in (bidx.at(t), lead_bass.at(t)) if b is not None]
        lo = max([COMP_FLOOR] + lows)
        # Where Guitar I keeps an inner voice going, Guitar II sits above it
        # (between it and the tune) when two chord tones fit there: four
        # layered voices instead of two guitars in one register.
        inner = lead_inner.pitches(t, t1)
        if inner and t1 is not None and pcs:
            above = max(inner) + 1
            if sum(1 for p in range(above, ceiling(t, t1) + 1) if p % 12 in pcs) >= 2:
                lo = max(lo, above)
        return lo

    def register(t0: int, t1: int, pcs: list[int]) -> tuple[int, int, bool]:
        """Guitar II's range for a chord tone in [t0, t1): (lo, hi, above_the_tune)."""
        lo, hi = floor(t0, t1, pcs), ceiling(t0, t1)
        taken = lead_poly.pitches(t0, t1)
        free = sum(1 for p in range(lo, hi + 1) if p % 12 in pcs and p not in taken)
        top = mel.max_between(t0, t1)
        if free < 2 and not song.percussive and top is not None and top <= 64:
            # A low tune with Guitar I's bass and harmony under it leaves no
            # room there: Guitar II rings a soft, high descant above it instead.
            return top + 3, min(COMP_CEIL + 7, top + 15), True
        return lo, hi, False

    def inner_motion(bar: Bar) -> bool:
        return sum(1 for n in lead_inner.notes if bar.start <= n.start < bar.end) >= 2

    for a, b in section_ranges(song):
        sec = [n for n in src if a <= song.bar_at(n.start) < b]
        pat = find_pattern(song, sec, range(a, b))
        for bi in range(a, b):
            bar = song.bars[bi]
            if bi in lead_chord_bars:
                # Guitar I strikes the chords here; Guitar II keeps the bass and, in
                # a soft song, rings a 12th/7th-fret harmonic chord tone over them.
                ch = song.chord_at(bar.start)
                if ch is not None and not song.percussive and energy_at(song, bar.start) < 0.55:
                    from .lead import natural_harmonic_pitches

                    pcs = chord_pcs(ch.root, ch.quality)
                    lead_here = {n.pitch for n in lead_notes if n.start <= bar.start < n.end}
                    bells = sorted((p for p in natural_harmonic_pitches(tuning, capo, (12, 7)) if p % 12 in pcs and p >= 64 and p not in lead_here), reverse=True)
                    for p in bells[:1]:
                        out.append(Note(bar.start, min(bar.end, ch.end) if ch.end > bar.start else bar.end, p, velocity=0.55, role="harmonic", priority=1.1, source="harmonics", tags={"harmonic_only"}))
                continue
            energy = energy_at(song, bar.start)
            label = section_label(song, bar.start)
            harp = (not song.percussive) and (energy < 0.45 or "intro" in label or "outro" in label)
            if pat is not None and len(pat.slots) >= 2 and bar.length == pat.length:
                ticks = [bar.start + t for t in pat.ticks() if t < bar.length]
                k_arp = 0
                for k, t in enumerate(ticks):
                    ch = song.chord_at(t)
                    if ch is None:
                        continue
                    nxt = ticks[k + 1] if k + 1 < len(ticks) else bar.end
                    ring = min(ch.end, bar.end) if ch.end > t else bar.end
                    pcs = chord_pcs(ch.root, ch.quality)
                    lo, hi, above = register(t, max(nxt, t + 1), pcs)
                    if hi - lo < 3:
                        continue
                    bass_here = bidx.at(t) is not None and bidx.at(t).start == t
                    taken = lead_poly.pitches(t)  # Guitar I already sounds these: pick other chord tones
                    if pat.block:
                        v = voicing(pcs, lo, hi, prev_voicing, 3 if energy > 0.35 else 2, avoid=taken)
                        prev_voicing = v or prev_voicing
                        for j, p in enumerate(v):
                            out.append(Note(t, max(nxt, ring), p, velocity=0.5 + 0.3 * energy, role="comp", priority=3.2 - 0.3 * j, source="comp", tags={"octave_ok"}))
                    else:
                        if bass_here and k == 0:
                            continue  # the bass speaks on the first slot of the arpeggio
                        v = (harp_voicing(pcs, tuning, capo, lo, hi, prev_voicing, taken) if harp or above else []) or voicing(pcs, lo, hi, prev_voicing, 3, avoid=taken)
                        if not v:
                            continue
                        prev_voicing = v
                        if len(v) < 2 and (t - bar.start) % bar.ticks_per_beat:
                            continue  # one free chord tone is not an arpeggio: sound it on the beats only
                        p = v[min(ARP_ORDER[k_arp % len(ARP_ORDER)], len(v) - 1)]
                        k_arp += 1
                        tags = ({"arp"} if above else {"octave_ok", "arp"}) | ({"prefer_harmonic"} if harp or above else set())
                        vel = (0.42 if above else 0.5) + 0.25 * energy
                        out.append(Note(t, ring, p, velocity=vel, role="comp", priority=2.8, source="comp", tags=tags))
                continue
            # Sustained source (pads/strings) or nothing: idiomatic pattern from the chords.
            # Under Guitar I's own moving inner voice, eighths rather than sixteenths.
            steps = pattern_steps(bar, bar.tempo, song.percussive, min(energy, 0.45) if inner_motion(bar) else energy)
            for ch in chord_segments(song, bar.start, bar.end):
                pcs = chord_pcs(ch.root, ch.quality)
                seg0, seg1 = max(bar.start, ch.start), min(bar.end, ch.end)
                lo, hi, above = register(seg0, seg1, pcs)
                if hi - lo < 5:
                    continue
                label = section_label(song, bar.start)
                harp = (not song.percussive) and (energy < 0.45 or "intro" in label or "outro" in label)
                v = (harp_voicing(pcs, tuning, capo, lo, hi, prev_voicing) if harp or above else []) or voicing(pcs, lo, hi, prev_voicing, 3)
                if not v:
                    continue
                prev_voicing = v
                for off, voice in steps:
                    t = bar.start + off
                    if not seg0 <= t < seg1:
                        continue
                    if len(v) < 2 and voice not in ("B", "A") and off % bar.ticks_per_beat:
                        continue  # one chord tone repeated is not a pattern: on the beats only
                    if voice in ("B", "A"):
                        if bidx.at(t) is not None and bidx.at(t).start == t:
                            continue  # the real bass line already speaks here
                        bnote = bidx.at(t)
                        if bnote is None:
                            continue
                        if voice == "A" and song.percussive:
                            fifth = (ch.root + 7) % 12
                            p = fit_bass(36 + fifth, lowest_pitch(tuning, capo))
                            if abs(p - bnote.pitch) > 9:
                                p = bnote.pitch
                        else:
                            p = bnote.pitch
                        out.append(Note(t, min(seg1, t + 2 * bar.ticks_per_beat), p, velocity=0.62, role="bass", priority=8.0, source="pattern"))
                        continue
                    if voice == "D":
                        pitches = v[-2:]
                    else:
                        idx = {"1": 0, "2": 1, "3": 2}[voice]
                        pitches = [v[min(idx, len(v) - 1)]]
                    ring_end = seg1 if not song.percussive else min(seg1, t + bar.ticks_per_beat)
                    taken = lead_poly.pitches(t)
                    pitches = [q for q in pitches if q not in taken]  # Guitar I already sounds it
                    for p in pitches:
                        tags = ({"arp"} if above else {"octave_ok", "arp"}) | ({"prefer_harmonic"} if harp or above else set())
                        vel = (0.45 if above else 0.55) + 0.2 * energy
                        out.append(Note(t, ring_end, p, velocity=vel, role="comp", priority=2.6, source="pattern", tags=tags))
    return resolve_collisions(out, lead_all, lambda n: n.role == "bass" and song.bar_at(n.start) in lead_chord_bars)


def resolve_collisions(notes: list[Note], lead: LineIndex, may_double=None) -> list[Note]:
    """Drop Guitar II notes that duplicate Guitar I's exact pitch at the same onset:
    two guitars striking one note add volume, not music (``may_double`` names
    the exceptions)."""
    lead_attacks = {(n.start, n.pitch) for n in lead.notes}
    seen = set()
    out = []
    for n in sorted(notes, key=lambda n: (n.start, -n.priority)):
        k = (n.start, n.pitch)
        if k in seen:
            continue
        if k in lead_attacks and not (may_double and may_double(n)):
            continue
        seen.add(k)
        out.append(n)
    return out
