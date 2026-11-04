"""Guitar I: the singing voice of the duet.

It carries the vocal melody (or the instrumental hook where nobody sings),
answers with the original fills in the melody's rests, thickens important
melody notes with chord tones underneath and adds harmonics where the song
is soft and open.
"""

from __future__ import annotations

from ..guitar import NATURAL_HARMONICS
from ..model import TPQ, Note, Song
from ..theory import chord_pcs
from .lines import LineIndex, best_octave_shift, gaps, monophonic, notes_between

MELODY_LO, MELODY_HI, MELODY_TARGET = 57, 86, 70
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
    for name, notes in song.parts.items():
        if name in exclude or name in ("drums", "bass", "vocals"):
            continue
        if not any(name.startswith(p) for p in HARMONY_PARTS):
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
    """How well a melody sits on the guitar after an octave shift (higher is better)."""
    if not pitches:
        return 0.0
    total = 0.0
    for p in pitches:
        q = p + shift
        if 57 <= q <= 79:
            total += 1.0
        elif 79 < q <= 84:
            total += 0.7
        elif q > 84:
            total -= 1.0
        elif 52 <= q < 57:
            total += 0.5
        else:
            total -= 0.6
    med = sorted(pitches)[len(pitches) // 2] + shift
    return total / len(pitches) - 0.006 * abs(med - 68)


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
    # Instrumental stretches: two bars or longer without singing get the hook.
    min_gap = 2 * song.bars[0].length if len(song.bars) > 2 else song.bars[0].length
    for g0, g1 in gaps(vocal, song.bars[0].start, song.end_tick, min_gap):
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


def support_notes(song: Song, melody: list[Note], density: float = 1.0) -> list[Note]:
    """Chord tones under important melody notes (3rds/6ths first)."""
    out: list[Note] = []
    mel_index = LineIndex(melody)
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
        for _ in range(n_support):
            best = None
            for iv, cost in SUPPORT_INTERVALS.items():
                p = top - iv
                if p < 52 or p % 12 not in pcs or p % 12 == m.pitch % 12 and iv != 12:
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
    nh = natural_harmonic_pitches(tuning, capo, (12, 7, 5))
    mel = LineIndex(melody)
    for bi, bar in enumerate(song.bars):
        label = section_label(song, bar.start)
        quiet = energy_at(song, bar.start) < 0.42
        if not (quiet or "intro" in label or "outro" in label or "bridge" in label):
            continue
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
        if long and phrase_end and (energy_at(song, m.start) < 0.55 or "outro" in label or "intro" in label):
            m.tags.add("prefer_harmonic")


def lead_part(song: Song, tuning, capo: int, density: float = 1.0, shift: int | None = None) -> list[Note]:
    melody = build_lead_line(song, shift)
    mark_harmonic_melody(song, melody)
    notes = list(melody)
    notes += support_notes(song, melody, density)
    notes += fills(song, melody)
    notes += harmonic_colour(song, melody, tuning, capo)
    return dedupe(notes)


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
