"""Turn a fingering solution into notated notes with expressive techniques."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..fingering import Realised, Solution
from ..model import Song


@dataclass
class PlacedNote:
    start: int
    end: int
    string: int
    fret: object  # int/float or "x"
    pitch: int | None
    role: str
    velocity: float
    harmonic: str = ""  # "", natural, artificial
    harmonic_interval: int = 0
    techs: list = field(default_factory=list)
    lh: str = ""
    rh: str = ""
    strum: bool = False
    sustain: bool = False
    source_id: int = 0
    tags: set = field(default_factory=set)


def placed_notes(sol: Solution, song: Song, tuning, capo: int) -> list[PlacedNote]:
    """Convert solver output; percussion carried by other notes is merged in."""
    out: list[PlacedNote] = []
    by_tick: dict[int, list[PlacedNote]] = {}
    carried: list[tuple[int, Realised]] = []
    for r in sol.notes:
        o = r.option
        if not o.places:
            if o.attach:
                carried.append((r.start, r))
            continue
        for k, pl in enumerate(o.places):
            pn = PlacedNote(
                start=r.start,
                end=max(r.end, r.start + 1),
                string=pl.string,
                fret=pl.fret,
                pitch=o.pitch,
                role=r.item.role,
                velocity=float(getattr(r.item.note, "velocity", 0.7)),
                lh=r.lh_finger if k == 0 else "",
                rh=r.rh_finger if k == 0 else "",
                strum=r.strum,
                source_id=r.item.id,
                tags=set(r.item.tags),
            )
            if pl.kind == "nh":
                pn.harmonic = "natural"
                pn.sustain = True
            elif pl.kind == "ah":
                pn.harmonic = "artificial"
                pn.harmonic_interval = 12
                pn.sustain = True
            elif pl.kind == "tap":
                pn.techs.append("RH")
            elif pl.kind == "dead":
                pn.fret = "x"
                pn.pitch = None
                pn.end = pn.start + 3
                pn.techs.extend(o.techs)
            if r.lh_tap and pl.kind == "fret":
                pn.techs.append("LH")
            out.append(pn)
            by_tick.setdefault(r.start, []).append(pn)
    for tick, r in carried:
        notes = [n for n in by_tick.get(tick, []) if n.fret != "x"]
        if not notes:
            continue
        tech = r.option.techs[0] if r.option.techs else ""
        if r.option.attach == "bass":
            target = [n for n in notes if n.role == "bass" and n.string >= 4]
        else:
            target = notes
        if not target:
            continue
        carrier = max(target, key=lambda n: n.string)
        if tech and tech not in carrier.techs:
            carrier.techs.append(tech)
    out.sort(key=lambda n: (n.start, n.string))
    return out


def beat_len(song: Song, tick: int) -> int:
    return song.bars[song.bar_at(tick)].ticks_per_beat


def add_expression(notes: list[PlacedNote], song: Song) -> None:
    """Legato, slides, vibrato, scoops/falls and accents on the melodic voice."""
    by_start: dict[int, list[PlacedNote]] = {}
    for n in notes:
        by_start.setdefault(n.start, []).append(n)
    starts = sorted(by_start)
    next_start = {s: (starts[i + 1] if i + 1 < len(starts) else None) for i, s in enumerate(starts)}
    lead = [n for n in notes if n.role in ("melody", "fill") and n.fret != "x"]
    lead.sort(key=lambda n: n.start)
    slides_in_bar: dict[int, int] = {}
    for a, b in zip(lead, lead[1:]):
        if a.string != b.string or a.harmonic or b.harmonic:
            continue
        if "RH" in a.techs or "RH" in b.techs:
            continue
        if next_start.get(a.start) != b.start:
            continue  # something else is attacked in between
        others = [n for n in by_start[b.start] if n is not b]
        if others:
            continue  # the right hand plucks anyway; keep it articulated
        legato = a.end >= b.start - 2
        dt = song.seconds_at(b.start) - song.seconds_at(a.start)
        if not isinstance(a.fret, (int, float)) or not isinstance(b.fret, (int, float)):
            continue
        d = b.fret - a.fret
        beat = beat_len(song, a.start)
        if legato and 1 <= abs(d) <= 4 and dt <= 0.34 and (d > 0 or b.fret >= 0):
            if d > 0:
                a.techs.append("H")
            elif a.fret > 0:
                a.techs.append("P")
            continue
        if legato and 2 <= abs(d) <= 7 and a.fret > 0 and b.fret > 0 and dt <= 0.6 and b.start - a.start <= beat:
            bar = song.bar_at(a.start)
            if slides_in_bar.get(bar, 0) >= 1:
                continue
            a.techs.append("LS" if abs(d) <= 4 else ("/" if d > 0 else "\\"))
            slides_in_bar[bar] = slides_in_bar.get(bar, 0) + 1
    for n in lead:
        if n.harmonic or not isinstance(n.fret, (int, float)):
            continue
        beat = beat_len(song, n.start)
        secs = song.seconds_at(n.end) - song.seconds_at(n.start)
        if n.role == "melody" and n.fret > 0 and "RH" not in n.techs and (n.end - n.start >= int(1.5 * beat)) and secs >= 0.85:
            if "V" not in n.techs:
                n.techs.append("V")
    # vocal scoops and falls captured by the transcription
    for n in lead:
        if n.pitch is None or not isinstance(n.fret, (int, float)):
            continue
        tags = n.tags
        if "scoop" in tags and n.fret >= 2 and not n.harmonic and "SI" not in n.techs:
            n.techs.append("SI")
        if "fall" in tags and n.fret >= 3 and not n.harmonic and "SO" not in n.techs and not any(t in n.techs for t in ("H", "P", "LS", "/", "\\")):
            n.techs.append("SO")
    # accents: syncopated strong hits in drum-driven songs
    if song.percussive:
        for n in notes:
            if n.fret == "x" or n.role not in ("melody", "comp", "support"):
                continue
            beat = beat_len(song, n.start)
            bar = song.bars[song.bar_at(n.start)]
            off = (n.start - bar.start) % beat
            if n.velocity >= 0.85 and off != 0 and "AC" not in n.techs:
                n.techs.append("AC")
