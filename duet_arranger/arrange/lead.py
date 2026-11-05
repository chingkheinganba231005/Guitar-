"""Guitar I: the singing voice of the duet, as a complete fingerstyle part.

It carries the vocal melody (or the instrumental hook where nobody sings)
the way a solo fingerstyle guitarist would: with its own bass on the strong
beats and chord changes, chord tones under important melody notes, inner
voices that keep the harmony moving while the tune holds, the original fills
in the melody's rests and harmonics where the song is soft and open.
"""

from __future__ import annotations

from ..guitar import NATURAL_HARMONICS
from ..model import TPQ, Note, Song
from ..theory import chord_pcs
from .lines import LineIndex, bassline, best_octave_shift, gaps, monophonic, notes_between

MELODY_LO, MELODY_HI, MELODY_TARGET = 57, 79, 68
HARMONY_PARTS = ("piano", "guitar", "other", "keys", "synth", "strings", "lead")

# Preferred intervals below the melody for a supporting voice (semitones -> cost)
SUPPORT_INTERVALS = {3: 0.0, 4: 0.0, 8: 0.15, 9: 0.15, 5: 0.5, 7: 0.55, 10: 0.9, 12: 1.0, 6: 1.6}


def beat_ticks_at(song: Song, tick: int) -> int:
    return song.bars[song.bar_at(tick)].ticks_per_beat if song.bars else TPQ


def is_strong(song: Song, tick: int) -> bool:
    b = song.bars[song.bar_at(tick)]
    off = tick - b.start
    if off == 0:
        return True
    half = b.length // 2
    return b.pulses % 2 == 0 and off == half


def energy_at(song: Song, tick: int) -> float:
    if not song.energy:
        return 0.5
    return song.energy[min(len(song.energy) - 1, song.bar_at(tick))]


def section_label(song: Song, tick: int) -> str:
    s = song.section_at_bar(song.bar_at(tick))
    return (s.label if s else "").lower()


def instrumental_line(song: Song, start: int, end: int, exclude: tuple = ()) -> list[Note]:
    """The most melodic top line among the harmony stems inside [start, end)."""
    best: list[Note] = []
    best_score = 0.0
    keep = set(main_harmony_parts(song))
    for name, notes in song.parts.items():
        if name in exclude or name in ("drums", "bass", "vocals"):
            continue
        if name not in keep:
            continue
        cand = monophonic([n for n in notes if start <= n.start < end], "high")
        if len(cand) < 2:
            continue
        pitches = [n.pitch for n in cand]
        motion = sum(1 for a, b in zip(pitches, pitches[1:]) if a != b)
        spread = max(pitches) - min(pitches)
        score = motion * (1 + min(spread, 12) / 12) + 0.3 * len(cand)
        if score > best_score:
            best, best_score = cand, score
    return best


def register_score(pitches: list[int], shift: int) -> float:
    """How well a melody sits on the guitar after an octave shift (higher is better).

    Up to about E5 the tune sings on the first five frets to the 12th; above
    that an acoustic thins out, so higher octaves pay for every note up there."""
    if not pitches:
        return 0.0
    total = 0.0
    for p in pitches:
        q = p + shift
        if 57 <= q <= 76:
            total += 1.0
        elif 76 < q <= 79:
            total += 0.4
        elif q > 79:
            total -= 1.5
        elif 52 <= q < 57:
            total += 0.5
        else:
            total -= 0.6
    med = sorted(pitches)[len(pitches) // 2] + shift
    return total / len(pitches) - 0.006 * abs(med - 66)


def melody_shift_candidates(song: Song, keep: int = 2) -> list[int]:
    """The most plausible octave shifts for the sung melody, best first."""
    pitches = [n.pitch for n in song.melody]
    if not pitches:
        return [0]
    scored = sorted(((register_score(pitches, s), s) for s in (-24, -12, 0, 12, 24)), reverse=True)
    best = scored[0][0]
    return [s for sc, s in scored[:keep] if sc >= best - 0.25]


def build_lead_line(song: Song, shift: int | None = None) -> list[Note]:
    """Vocal melody where there is singing, the instrumental hook elsewhere."""
    vocal = [n.copy(role="melody", priority=100.0, source=n.source or "vocals") for n in monophonic(song.melody, "loud")]
    if vocal:
        if shift is None:
            shift = melody_shift_candidates(song, 1)[0]
        for n in vocal:
            n.pitch += shift
    lead = list(vocal)
    if not song.bars:
        return lead
    # Instrumental stretches: two bars or longer without singing get the hook,
    # unless the band only strikes chords there (Guitar I plays those instead).
    chord_ticks = {(song.bars[a].start, song.bars[b - 1].end) for a, b in chordal_regions(song, vocal)}
    for g0, g1 in instrumental_gaps(song, vocal):
        if any(c0 <= g0 and g1 <= c1 for c0, c1 in chord_ticks):
            continue
        hook = instrumental_line(song, g0, g1)
        if not hook:
            continue
        pitches = [n.pitch for n in hook]
        shift = max((register_score(pitches, s) + (0.02 if s == 0 else 0.0), s) for s in (-24, -12, 0, 12, 24))[1]
        for n in hook:
            lead.append(n.copy(pitch=n.pitch + shift, role="melody", priority=100.0, tags=set(n.tags) | {"instrumental"}))
    lead.sort(key=lambda n: n.start)
    out = monophonic(lead, "high")
    for n in out:
        n.role = "melody"
        n.priority = 100.0
    return out


def instrumental_gaps(song: Song, vocal: list[Note]) -> list[tuple[int, int]]:
    if not song.bars:
        return []
    min_gap = 2 * song.bars[0].length if len(song.bars) > 2 else song.bars[0].length
    return gaps(vocal, song.bars[0].start, song.end_tick, min_gap)


def main_harmony_parts(song: Song) -> list[str]:
    """Harmony stems that carry the music.

    Separation leaks a little of every instrument into the other stems (a
    song with no guitar still has a "guitar" stem); transcribed, that leakage
    is a scatter of off-beat notes. For rhythm and line decisions only stems
    at least 60% as loud as the loudest harmony stem count. (MIDI has no
    leakage: every harmony part counts.)"""
    names = [n for n in song.parts if any(n.startswith(p) for p in HARMONY_PARTS) and n not in ("vocals", "bass", "drums")]
    rms = (song.notes_info or {}).get("stem_rms") or {}
    levels = {n: rms.get(n) for n in names if rms.get(n) is not None}
    if not levels:
        return names
    loud = max(levels.values()) or 1e-9
    return [n for n in names if levels.get(n, loud) >= 0.6 * loud]


def harmony_notes(song: Song) -> list[Note]:
    keep = set(main_harmony_parts(song))
    return [n for name, ns in song.parts.items() if name in keep for n in ns]


def chordal_regions(song: Song, vocal: list[Note]) -> list[tuple[int, int]]:
    """Bar ranges without singing where the band strikes chords rather than a line."""
    from .patterns import chordal

    harm = harmony_notes(song)
    out = []
    for g0, g1 in instrumental_gaps(song, vocal):
        a = song.bar_at(g0)
        if song.bars[a].start < g0:
            a += 1
        b = song.bar_at(max(g0, g1 - 1)) + 1
        if b - a < 1:
            continue
        if g1 < song.bars[b - 1].end:
            b -= 1
        if b - a >= 1 and chordal(song, [n for n in harm if song.bars[a].start <= n.start < song.bars[b - 1].end], range(a, b)):
            # include a pickup bar before the region if it is also instrumental
            if a == 1 and song.bars[0].length < song.bars[1].length and not any(n.start < song.bars[1].start for n in vocal):
                a = 0
            out.append((a, b))
    return out


def lead_chords(song: Song, regions: list[tuple[int, int]], tuning, capo: int) -> list[Note]:
    """Guitar I strikes the band's chords on the section's pattern and lets them ring."""
    from .accomp import fit_bass, lowest_pitch, place_in_range, voicing
    from .patterns import find_pattern

    harm = harmony_notes(song)
    out: list[Note] = []
    prev = None
    lo_string = lowest_pitch(tuning, capo)
    from .patterns import section_ranges

    from .patterns import texture_runs

    pieces = []  # one groove per section, and per run of bars that strike alike
    for a, b in regions:
        for s0, s1 in section_ranges(song):
            lo, hi = max(a, s0), min(b, s1)
            if hi > lo:
                pieces += texture_runs(song, harm, lo, hi)
    for a, b in pieces:
        notes = [n for n in harm if song.bars[a].start <= n.start < song.bars[b - 1].end]
        pat = find_pattern(song, notes, range(a, b))
        for bi in range(a, b):
            bar = song.bars[bi]
            if pat is not None and bar.length == pat.length:
                ticks = [bar.start + t for t in pat.ticks() if t < bar.length]
            else:
                ticks = [bar.start]
            last_chord = None
            for k, t in enumerate(ticks):
                ch = song.chord_at(t)
                if ch is None:
                    continue
                nxt = ticks[k + 1] if k + 1 < len(ticks) else bar.end
                tops = [n.pitch for n in notes if abs(n.start - t) <= 6]
                top = place_in_range(max(tops), 62, 76) if tops else None
                pcs = chord_pcs(ch.root, ch.quality)
                v = voicing(pcs, 55, 76, prev, 3, prefer_top=top)
                if not v:
                    continue
                prev = v
                ring = min(ch.end, bar.end) if ch.end > t else nxt
                vel = 0.55 + 0.3 * energy_at(song, t)
                for j, p in enumerate(reversed(v)):
                    out.append(Note(t, max(nxt, ring), p, velocity=vel, role="chord", priority=7.0 if j == 0 else 3.5, source="chords", tags={"octave_ok"}))
                if last_chord is None or (ch.root, ch.quality) != last_chord:
                    root = ch.bass if ch.bass is not None else ch.root
                    bp = fit_bass(36 + root, lo_string)
                    if bp < min(v) - 2:
                        out.append(Note(t, max(nxt, ring), bp, velocity=vel, role="bass", priority=6.0, source="chords"))
                last_chord = (ch.root, ch.quality)
    return out


def held_spans(song: Song) -> list[tuple[int, int]]:
    """Moments where the singer holds a long note and the band holds the chord
    under it ("one step clo-ser~"): no new piano/guitar/bass notes and no
    drum hits until the next sung note. Both guitars strike the chord at the
    start of the note and let it ring; nobody keeps the rhythm going.
    Returns [start, end) tick spans."""
    import bisect

    if not song.bars or not song.melody:
        return []
    keep = set(main_harmony_parts(song))

    def real(n: Note) -> bool:  # not a short, quiet blip of leakage
        return n.end - n.start >= TPQ // 4 or n.velocity >= 0.35

    onsets = sorted({n.start for name, ns in song.parts.items() if name in keep or name == "bass" for n in ns if real(n)} | {n.start for n in song.bass if real(n)})
    drums = sorted(h.start for h in song.hits if h.kind in ("kick", "snare"))
    vocal = sorted(monophonic(song.melody, "loud"), key=lambda n: n.start)
    out: list[tuple[int, int]] = []
    for i, m in enumerate(vocal):
        beat = beat_ticks_at(song, m.start)
        if m.end - m.start < 2 * beat:
            continue  # a held note: at least two beats (a full bar of 6/8)
        nxt = vocal[i + 1].start if i + 1 < len(vocal) else song.end_tick
        bar_len = song.bars[song.bar_at(m.start)].length
        end = min(nxt, max(m.end, m.start + 2 * beat), m.start + 2 * bar_len)
        lo = m.start + TPQ // 4
        k = bisect.bisect_left(onsets, lo)
        inside = [t for t in onsets[k:] if t < m.end - TPQ // 4]
        if inside:
            continue  # the band keeps moving under the note
        j = bisect.bisect_left(drums, lo)
        if j < len(drums) and drums[j] < m.end - TPQ // 4:
            continue
        changes = [c for c in song.chords if m.start < c.start < m.end - TPQ // 4]
        if changes:
            continue
        # the hold lasts until the band (or the singer) moves again
        k2 = bisect.bisect_left(onsets, m.end - TPQ // 4)
        if k2 < len(onsets):
            end = min(end, max(m.end, onsets[k2]))
        out.append((m.start, end))
    return out


def in_hold(spans: list[tuple[int, int]], t: int) -> bool:
    """Is ``t`` strictly inside a held moment (after its first strike)?"""
    return any(a < t < b for a, b in spans)


def bass_ticks(song: Song, bar, density: float = 1.0) -> list[tuple[int, float]]:
    """Where a solo fingerstyle bass speaks in a bar, with its priority.

    The downbeat always; the bar's second strong beat (beat 3 in 4/4, beat 4
    in 6/8 and 12/8's third pulse) unless the texture is thinned; and every
    chord change."""
    out = {bar.start: 9.0}
    tpb, pulses = bar.ticks_per_beat, bar.pulses
    if density >= 0.8:
        mid = None
        if pulses in (4, 6):
            mid = bar.start + bar.length // 2
        elif pulses == 2 and tpb != TPQ:
            mid = bar.start + tpb
        if mid is not None:
            out[mid] = 5.0
    for c in song.chords:
        if bar.start < c.start < bar.end:
            out[c.start] = max(out.get(c.start, 0.0), 8.0)
    return sorted(out.items())


def lead_bass(song: Song, melody: list[Note], tuning, capo: int, skip_bars: set, density: float = 1.0) -> list[Note]:
    """Guitar I's own bass, so it stands as a complete fingerstyle part.

    The real bass note where it is a chord tone, else the chord's bass, on
    the low strings; it rings until the next bass note or the chord change."""
    from .accomp import fit_bass, lowest_pitch

    if not song.bars or not song.chords:
        return []
    lo = lowest_pitch(tuning, capo)
    holds = held_spans(song)
    src = LineIndex(bassline(song.bass)) if song.bass else None
    mel = LineIndex(melody)
    out: list[tuple[Note, int]] = []
    for bi, bar in enumerate(song.bars):
        if bi in skip_bars:
            continue
        for t, prio in bass_ticks(song, bar, density):
            if in_hold(holds, t):
                continue
            ch = song.chord_at(t)
            if ch is None:
                continue
            pcs = set(chord_pcs(ch.root, ch.quality))
            want = ch.bass if ch.bass is not None else ch.root
            p = fit_bass(36 + want, lo)
            if src is not None:
                n = src.at(t)
                if n is None or n.start < t - TPQ:
                    nxt = src.next_after(t - 1)
                    n = nxt if nxt is not None and nxt.start - t <= TPQ // 4 else n
                # the real bass, in its own octave where the guitar reaches it
                if n is not None and n.pitch % 12 in (pcs | {want}):
                    p = fit_bass(n.pitch, lo)
            m = mel.at(t)
            if m is None:
                m = mel.next_after(t)
                m = m if m is not None and m.start - t < beat_ticks_at(song, t) else None
            if m is not None and m.pitch - p < 7:
                if p - 12 >= lo:
                    p -= 12
                else:
                    continue
            vel = 0.58 + 0.25 * energy_at(song, t)
            out.append((Note(t, t + TPQ, p, velocity=vel, role="bass", priority=prio * density, source="bass"), ch.end))
    out.sort(key=lambda x: x[0].start)
    notes = [n for n, _ in out]
    for k, (n, ch_end) in enumerate(out):
        nxt = notes[k + 1].start if k + 1 < len(notes) else song.end_tick
        bar_len = song.bars[song.bar_at(n.start)].length
        n.end = max(n.start + 6, min(nxt, ch_end, n.start + bar_len))
    return notes


def inner_ticks(song: Song, bar, density: float) -> list[tuple[int, bool]]:
    """Felt beats of the bar, plus the eighths between them where the meter
    lives in them (6/8, 9/8, 12/8) or the beat is very slow: (tick, on_the_beat).

    The finer motion of a simple-meter song is Guitar II's to add."""
    tpb = bar.ticks_per_beat
    sub = None
    if density >= 0.9:
        if tpb != TPQ:
            sub = tpb // 3
        elif 60.0 / max(1.0, bar.tempo) >= 1.0:
            sub = tpb // 2
    out = []
    for k in range(bar.pulses):
        b = bar.start + k * tpb
        out.append((b, True))
        if sub:
            out += [(b + j * sub, False) for j in range(1, tpb // sub)]
    return [(t, on) for t, on in out if t < bar.end]


def lead_inner(song: Song, melody: list[Note], fill_notes: list[Note], bass: list[Note], skip_bars: set, avoid: set, density: float = 1.0) -> list[Note]:
    """Inner chord tones where the melody holds or rests, the way a solo
    fingerstyle guitar keeps the harmony moving under a long note."""
    out: list[Note] = []
    if not song.bars:
        return out
    mel = LineIndex(melody)
    bidx = LineIndex(bass)
    attacks = sorted({n.start for n in melody} | {n.start for n in fill_notes})
    holds = held_spans(song)
    import bisect

    prev: int | None = None
    for bi, bar in enumerate(song.bars):
        if bi in skip_bars:
            prev = None
            continue
        tpb = bar.ticks_per_beat
        ticks = inner_ticks(song, bar, density)
        sub = min((b - a for (a, _), (b, _) in zip(ticks, ticks[1:])), default=tpb)
        last_here: int | None = None
        for t, on_beat in ticks:
            if t in avoid or in_hold(holds, t):
                continue
            ch = song.chord_at(t)
            if ch is None:
                continue
            # no tune note starting on this tick, nor (between beats) right around it
            reach = 0 if on_beat else sub - 1
            i = bisect.bisect_left(attacks, t - reach)
            if i < len(attacks) and attacks[i] <= t + reach:
                continue
            m = mel.at(t)
            if m is not None:
                ceiling = m.pitch - 3
            else:
                pm = mel.prev_before(t)
                ceiling = min(69, pm.pitch - 3) if pm is not None and t - pm.end < bar.length else 67
                if not on_beat:
                    continue  # in a rest only the beats speak; the fills answer in between
            b = bidx.at(t)
            floor = max(52, b.pitch + 5) if b is not None else 52
            if ceiling < floor:
                continue
            pcs = chord_pcs(ch.root, ch.quality)
            bass_pc = b.pitch % 12 if b is not None else (ch.bass if ch.bass is not None else ch.root)
            best = None
            for p in range(floor, ceiling + 1):
                if p % 12 not in pcs:
                    continue
                cost = 0.12 * abs(p - prev) if prev is not None else 0.05 * abs(p - 60)
                if p % 12 == bass_pc:
                    cost += 0.5
                if m is not None and p % 12 == m.pitch % 12:
                    cost += 0.7
                if last_here is not None and p == last_here and not on_beat:
                    cost += 1.4  # between the beats an arpeggio moves
                if best is None or cost < best[1]:
                    best = (p, cost)
            if best is None:
                continue
            p = best[0]
            prev = last_here = p
            end = min(ch.end, t + tpb, bar.end)
            vel = 0.42 + 0.22 * energy_at(song, t)
            prio = (2.2 if on_beat else 1.4) * density
            out.append(Note(t, max(t + 6, end), p, velocity=vel, role="inner", priority=prio, source="harmony"))
    return out


def support_notes(song: Song, melody: list[Note], density: float = 1.0, bass: list[Note] | None = None) -> list[Note]:
    """Chord tones under important melody notes (3rds/6ths first)."""
    out: list[Note] = []
    mel_index = LineIndex(melody)
    bidx = LineIndex(bass or [])
    last_support_tick = -10**9
    for m in melody:
        ch = song.chord_at(m.start)
        if ch is None:
            continue
        beat = beat_ticks_at(song, m.start)
        dur = m.end - m.start
        strong = is_strong(song, m.start)
        changed = ch.start == m.start or (m.start - ch.start) < beat // 2
        if not (strong or changed or dur >= beat):
            continue
        if m.start - last_support_tick < beat and not changed:
            continue
        pcs = chord_pcs(ch.root, ch.quality)
        if m.pitch % 12 not in pcs and dur < 2 * beat:
            continue
        energy = energy_at(song, m.start)
        n_support = 1
        if density >= 1.0 and energy > 0.62 and dur >= beat and (strong or changed):
            n_support = 2
        # how long the support may ring: until the chord changes, at most two beats
        end = min(ch.end, m.start + 2 * beat)
        nxt = mel_index.next_after(m.start)
        if nxt is not None and nxt.start - m.start >= beat:
            end = min(end, nxt.start + beat)
        top = m.pitch
        chosen = []
        b = bidx.at(m.start)
        floor = max(52, b.pitch + 5) if b is not None else 52
        for _ in range(n_support):
            best = None
            for iv, cost in SUPPORT_INTERVALS.items():
                p = top - iv
                if p < floor or p % 12 not in pcs or p % 12 == m.pitch % 12 and iv != 12:
                    continue
                if any(abs(p - c) < 3 for c in chosen):
                    continue
                if best is None or cost < best[1]:
                    best = (p, cost)
            if best is None:
                break
            chosen.append(best[0])
            top = best[0]
        for k, p in enumerate(chosen):
            out.append(
                Note(
                    m.start,
                    max(m.start + beat // 2, end),
                    p,
                    velocity=m.velocity * 0.75,
                    role="support",
                    priority=2.6 - 0.6 * k,
                    source="harmony",
                    tags={"octave_ok"},
                )
            )
        if chosen:
            last_support_tick = m.start
    return out


def fills(song: Song, melody: list[Note]) -> list[Note]:
    """Original instrumental answers inside the melody's rests (inside sung sections)."""
    out: list[Note] = []
    if not song.bars or not melody:
        return out
    beat = song.bars[0].ticks_per_beat
    bar_len = song.bars[0].length
    for g0, g1 in gaps(melody, melody[0].start, melody[-1].end, int(1.5 * beat)):
        if g1 - g0 >= 2 * bar_len:
            continue  # long gaps are handled as instrumental hooks
        line = instrumental_line(song, g0, g1)
        line = [n for n in line if n.start >= g0 and n.start < g1 - beat // 4]
        if len(line) < 2:
            continue
        prev = max((n for n in melody if n.end <= g0 + 1), key=lambda n: n.end, default=None)
        target = prev.pitch if prev else MELODY_TARGET
        shift = best_octave_shift([n.pitch for n in line], MELODY_LO - 3, MELODY_HI, target)
        for n in line:
            out.append(n.copy(pitch=n.pitch + shift, end=min(n.end, g1), role="fill", priority=6.0, tags=set(n.tags) | {"octave_ok"}))
    return out


def natural_harmonic_pitches(tuning, capo: int, nodes=(12, 7, 5, 19)) -> dict[int, list[tuple[int, float]]]:
    """pitch -> [(string, node)] for natural harmonics in this tuning."""
    out: dict[int, list] = {}
    for s in range(6):
        for node in nodes:
            semi, _ = NATURAL_HARMONICS[node]
            out.setdefault(tuning[s] + capo + semi, []).append((s + 1, node))
    return out


def harmonic_colour(song: Song, melody: list[Note], tuning, capo: int) -> list[Note]:
    """Bell-like natural-harmonic chord tones where the song is soft and the melody rests.

    Only used for melodic (non drum-driven) songs, in intros/outros, quiet
    passages and on long melody notes; the solver drops any it cannot place.
    """
    out: list[Note] = []
    if song.percussive or not song.bars:
        return out
    nh = natural_harmonic_pitches(tuning, capo, (12, 7))
    mel = LineIndex(melody)
    for bi, bar in enumerate(song.bars):
        label = section_label(song, bar.start)
        quiet = energy_at(song, bar.start) < 0.42
        sung = any(bar.start <= n.start < bar.end for n in melody if "instrumental" not in n.tags)
        if not ((quiet and not sung) or "intro" in label or "outro" in label or "bridge" in label):
            continue  # under a sung verse harmonic chords only blur the tune
        ch = song.chord_at(bar.start)
        if ch is None:
            continue
        pcs = chord_pcs(ch.root, ch.quality)
        here = mel.at(bar.start)
        attack = here is not None and here.start == bar.start
        if attack:
            continue  # the melody note itself speaks on the downbeat
        ceiling = (here.pitch - 1) if here is not None else 200
        tones = sorted(p for p in nh if p % 12 in pcs and 64 <= p < ceiling)
        if not tones:
            continue
        pick = tones[-2:] if len(tones) >= 2 else tones
        end = min(bar.end, ch.end, bar.start + bar.length)
        for p in pick:
            out.append(Note(bar.start, end, p, velocity=0.6, role="harmonic", priority=1.1, source="harmonics", tags={"harmonic_only"}))
    return out


def mark_harmonic_melody(song: Song, melody: list[Note]) -> None:
    """Ask the solver to try harmonics on long, exposed melody notes in soft songs."""
    if song.percussive:
        return
    for i, m in enumerate(melody):
        beat = beat_ticks_at(song, m.start)
        nxt = melody[i + 1] if i + 1 < len(melody) else None
        phrase_end = nxt is None or nxt.start - m.end >= beat
        long = m.end - m.start >= 2 * beat
        label = section_label(song, m.start)
        # the sung tune keeps its full fretted voice; harmonics only colour a
        # long last note of an intro or outro line, or the song's very last note
        if long and phrase_end and ("outro" in label or "intro" in label or nxt is None):
            m.tags.add("prefer_harmonic")


def lead_part(song: Song, tuning, capo: int, density: float = 1.0, shift: int | None = None) -> list[Note]:
    melody = build_lead_line(song, shift)
    mark_harmonic_melody(song, melody)
    vocal = [n for n in melody if "instrumental" not in n.tags]
    regions = chordal_regions(song, vocal)
    chord_bars = {bi for a, b in regions for bi in range(a, b)}
    notes = list(melody)
    # A complete fingerstyle part: melody on top, its own bass below and the
    # harmony in between, so Guitar I stands on its own and Guitar II is free
    # to add groove and colour.
    fl = fills(song, melody)
    bass = lead_bass(song, melody, tuning, capo, chord_bars, density)
    bells = [n for n in harmonic_colour(song, melody, tuning, capo) if song.bar_at(n.start) not in chord_bars]
    notes += support_notes(song, melody, density, bass)
    notes += fl
    notes += bass
    notes += lead_inner(song, melody, fl, bass, chord_bars, {n.start for n in bells}, density)
    notes += lead_chords(song, regions, tuning, capo)
    notes += bells
    sustain_melody(song, melody)
    return dedupe(notes)


def sustain_melody(song: Song, melody: list[Note]) -> None:
    """A singer holds each note into the next one; the transcription often
    ends it early. The tune rings on until the next tune note (up to two
    beats past its written end, and through a held moment), so the fingering
    keeps that string sounding: no fill, inner voice or bass may cut it short,
    on its string or by moving the hand to a far position while a fretted
    tune note still rings (a fill that would is handed to Guitar II)."""
    holds = held_spans(song)
    for i, m in enumerate(melody):
        if "instrumental" in m.tags:
            continue
        beat = beat_ticks_at(song, m.start)
        nxt = melody[i + 1].start if i + 1 < len(melody) else song.end_tick
        end = max(m.end, min(nxt, m.end + 2 * beat))
        for a, b in holds:
            if a == m.start or a < m.start < b:
                end = max(end, min(nxt, b))
        m.end = end


def dedupe(notes: list[Note]) -> list[Note]:
    """Remove same-pitch duplicates at the same onset, keeping the most important."""
    best: dict[tuple, Note] = {}
    for n in notes:
        k = (n.start, n.pitch)
        if k not in best or n.priority > best[k].priority:
            best[k] = n
    return sorted(best.values(), key=lambda n: (n.start, -n.priority))


def melody_of(notes: list[Note]) -> list[Note]:
    return [n for n in notes if n.role == "melody"]


def ceiling_index(notes: list[Note]) -> LineIndex:
    return LineIndex(melody_of(notes))


__all__ = [
    "build_lead_line",
    "lead_part",
    "support_notes",
    "fills",
    "harmonic_colour",
    "natural_harmonic_pitches",
    "notes_between",
]
