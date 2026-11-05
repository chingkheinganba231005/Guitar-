"""Write an Atelier (version 2, multi-track) project for the duet."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..arrange.duet import Arrangement, GuitarArrangement
from ..arrange.techniques import PlacedNote
from ..model import TPQ, Song
from ..quantize import Grid
from ..theory import prefers_flats

STRING_DYNAMICS = [100, 85, 65, 45, 30, 20]  # Atelier's balanced defaults, string 1..6
ROLE_GAIN = {"melody": 1.18, "fill": 1.0, "support": 0.78, "inner": 0.72, "octave": 0.95, "comp": 0.8, "bass": 1.0, "harmonic": 1.0, "perc": 1.0, "resonance": 0.7}

# (ticks, den, dots) for straight durations, longest first
STRAIGHT = [(96, 1, 0), (72, 2, 1), (48, 2, 0), (36, 4, 1), (24, 4, 0), (18, 8, 1), (12, 8, 0), (9, 16, 1), (6, 16, 0), (3, 32, 0)]


def split_straight(offset: int, length: int, tpb: int) -> list[tuple[int, int, int]]:
    """Split a duration that starts ``offset`` ticks into the bar into notatable values."""
    pieces = []
    o, rem = offset, length
    guard = 0
    while rem > 0 and guard < 64:
        guard += 1
        for L, den, dots in STRAIGHT:
            if L > rem:
                continue
            if L >= tpb:
                if o % tpb or (tpb != TPQ and L % tpb):
                    continue
            else:
                unit = 6 if L in (12, 18) else 3
                if tpb == 36 and L == 24:
                    unit = 12
                if o % unit:
                    continue
                if o // tpb != (o + L - 1) // tpb:
                    continue
            pieces.append((L, den, dots))
            o += L
            rem -= L
            break
        else:
            # Odd remainder (never expected on the 3-tick grid): use 32nds.
            pieces.append((3, 32, 0))
            o += 3
            rem -= 3
    return pieces


@dataclass
class Piece:
    start: int
    length: int
    den: int
    dots: int
    ratio: tuple


def bar_pieces(song: Song, bi: int, boundaries: set[int], grid: Grid) -> list[Piece]:
    bar = song.bars[bi]
    tpb = bar.ticks_per_beat
    bounds = {bar.start, bar.end} | {t for t in boundaries if bar.start <= t < bar.end}
    triplet_beats = {}
    for k in range(bar.pulses):
        b0 = bar.start + k * tpb
        b1 = min(bar.end, b0 + tpb)
        gi = grid.beat_index(b0)
        if tpb == TPQ and grid.is_triplet(gi) and b1 - b0 == TPQ:
            inner = [t for t in bounds if b0 < t < b1]
            if inner:
                step = 8 if all((t - b0) % 8 == 0 for t in inner) else 4
                triplet_beats[b0] = step
                bounds |= {b0 + j * step for j in range(TPQ // step + 1)}
    pts = sorted(bounds)
    pieces: list[Piece] = []
    for a, b in zip(pts, pts[1:]):
        beat_start = bar.start + ((a - bar.start) // tpb) * tpb
        if beat_start in triplet_beats and a < beat_start + tpb:
            step = triplet_beats[beat_start]
            if step == 8:
                pieces.append(Piece(a, b - a, 8, 0, (3, 2)))
            else:
                pieces.append(Piece(a, b - a, 16, 0, (6, 4)))
            continue
        for L, den, dots in split_straight(a - bar.start, b - a, tpb):
            pieces.append(Piece(a, L, den, dots, (1, 1)))
            a += L
    return pieces


def atelier_velocity(n: PlacedNote) -> int:
    if n.harmonic:
        v = 127 * (0.75 + 0.45 * n.velocity)
    else:
        base = STRING_DYNAMICS[n.string - 1]
        v = base * (0.6 + 0.85 * n.velocity) * ROLE_GAIN.get(n.role, 1.0)
    return int(max(1, min(200, round(v))))


def note_json(n: PlacedNote) -> dict:
    fret = n.fret
    if isinstance(fret, float) and fret.is_integer():
        fret = int(fret)
    d = {
        "string": n.string,
        "fret": fret,
        "sustain": bool(n.sustain),
        "tech": list(dict.fromkeys(n.techs)),
        "harmonic": n.harmonic,
        "grace": None,
        "velocity": atelier_velocity(n),
        "tone": "",
    }
    if n.harmonic and n.harmonic != "natural":
        d["harmonicInterval"] = n.harmonic_interval or 12
    if n.lh:
        d["lh"] = n.lh
    if n.rh:
        d["rh"] = n.rh
    return d


def significant_silences(song: Song, notes: list[PlacedNote]) -> list[tuple[int, int]]:
    """Stretches of at least a bar (and two beats) where this guitar has nothing to play."""
    spans = sorted((n.start, n.end) for n in notes if n.fret != "x")
    out = []
    cur = spans[0][1] if spans else 0
    for a, b in spans[1:]:
        if a > cur:
            bar = song.bars[song.bar_at(cur)]
            if a - cur >= max(bar.length, 2 * bar.ticks_per_beat):
                out.append((cur, a))
        cur = max(cur, b)
    return out


def apply_let_ring(song: Song, notes: list[PlacedNote]) -> list[tuple[int, int]]:
    """Fingerstyle guitar rings: every note lets ring, except the ones that would
    ring into a real silence, which stop at their written length."""
    for n in notes:
        n.sustain = n.fret != "x"
    gaps = significant_silences(song, notes)
    for g0, _ in gaps:
        last: dict[int, PlacedNote] = {}
        for n in notes:
            if n.start < g0 and n.fret != "x":
                if n.string not in last or n.start >= last[n.string].start:
                    last[n.string] = n
        for n in last.values():
            if not n.harmonic:
                n.sustain = False
    return gaps


def track_bars(song: Song, g: GuitarArrangement, track_index: int, grid: Grid, flats: bool) -> list[dict]:
    notes = sorted(g.placed, key=lambda n: (n.start, n.string))
    rests = apply_let_ring(song, notes)
    rest_starts = {g0 for g0, _ in rests}
    by_bar: dict[int, list[PlacedNote]] = {}
    for n in notes:
        by_bar.setdefault(song.bar_at(n.start), []).append(n)
    # silence starts: ticks where nothing of this guitar sounds any more
    events_out = []
    sounding: dict[int, tuple[PlacedNote, dict]] = {}  # string -> (note, last json)
    chord_starts = {c.start: c for c in song.chords}
    chord_ticks = sorted(chord_starts)
    eid = 0
    last_tempo = None
    last_meter = None
    section_at = {s.start_bar: s.label for s in song.sections}
    strum_ticks = set(g.solution.strum_ticks) | {n.start for n in notes if n.strum}
    prev_event_id = None
    chord_done = set()
    for bi, bar in enumerate(song.bars):
        bar_notes = by_bar.get(bi, [])
        attacks = {n.start for n in bar_notes}
        # Events start at attacks, and at the start of a real silence (a rest).
        silence = {t for t in rest_starts if bar.start < t < bar.end}
        pieces = bar_pieces(song, bi, attacks | silence, grid)
        events = []
        for pc in pieces:
            eid += 1
            ev_id = f"g{track_index}e{eid}"
            ev = {"id": ev_id, "den": pc.den, "dots": pc.dots, "ratio": list(pc.ratio), "notes": [], "above": "", "below": "", "strum": ""}
            here = [n for n in bar_notes if n.start == pc.start]
            used = set()
            for n in here:
                if n.string in used:
                    continue
                used.add(n.string)
                j = note_json(n)
                ev["notes"].append(j)
                if n.fret != "x":
                    sounding[n.string] = (n, j)
                else:
                    sounding.pop(n.string, None)
            for s, (n, j) in list(sounding.items()):
                if s in used:
                    continue
                if n.sustain:
                    # A ringing note needs no tie once anything new is struck; inside
                    # its own span the tie keeps split note values from reading as rests.
                    keep = pc.start not in attacks and pc.start not in silence and prev_event_id is not None
                else:
                    keep = n.start < pc.start and n.end >= pc.start + max(1, pc.length // 2) and prev_event_id is not None
                if keep:
                    t = dict(j)
                    t["tie"] = True
                    t["tieFrom"] = prev_event_id
                    t["grace"] = None
                    ev["notes"].append(t)
                    used.add(s)
                    sounding[s] = (n, t)
                else:
                    del sounding[s]
            if pc.start in strum_ticks and len([x for x in ev["notes"] if not x.get("tie")]) >= 2:
                ev["strum"] = "down"
            c = chord_starts.get(pc.start)
            if c is None:
                # a chord that began inside a piece (or a tied-over one): mark on first piece that holds it
                for t in chord_ticks:
                    if pc.start <= t < pc.start + pc.length and t not in chord_done:
                        c = chord_starts[t]
                        break
            if c is not None and c.start not in chord_done:
                ev["chord"] = c.name(flats)
                chord_done.add(c.start)
            ev["notes"].sort(key=lambda x: x["string"])
            events.append(ev)
            prev_event_id = ev_id
        b = {
            "id": f"bar{bi + 1}",
            "section": section_at.get(bi, ""),
            "meter": None,
            "tempo": None,
            "tone": "",
            "above": "",
            "below": "",
            "events": events,
        }
        if last_meter is not None and bar.meter != last_meter:
            b["meter"] = list(bar.meter)
        last_meter = bar.meter
        tempo = float(round(bar.tempo))
        if last_tempo is not None and abs(tempo - last_tempo) >= 1.0:
            b["tempo"] = tempo
            last_tempo = tempo
        elif last_tempo is None:
            last_tempo = tempo
        events_out.append(b)
    return events_out


def to_atelier(arr: Arrangement, title: str | None = None, artist: str | None = None) -> dict:
    song = arr.song
    grid = Grid(song)
    flats = prefers_flats(song.key_tonic, song.key_mode)
    tracks = []
    pans = [-0.28, 0.28, 0.0, 0.0]
    for ti, g in enumerate(arr.guitars):
        bars = track_bars(song, g, ti, grid, flats)
        if bars and not bars[0]["section"]:
            bars[0]["section"] = "Intro" if not song.sections else bars[0]["section"]
        label = "Lead · melody, bass & harmony" if g.role == "lead" else ("Accompaniment · harmony, bass groove & percussion" if song.percussive else "Accompaniment · harmony & bass groove")
        if bars:
            bars[0]["above"] = f"{g.tuning.name}{' · Capo ' + str(g.capo) if g.capo else ''} · let ring throughout"
        tracks.append(
            {
                "id": f"track{ti + 1}",
                "name": g.name,
                "role": label,
                "tuning": list(g.tuning.pitches),
                "capo": g.capo,
                "tone": "steel",
                "volume": 1.0 if g.role == "lead" else 0.9,
                "pan": pans[ti] if ti < len(pans) else 0.0,
                "mute": False,
                "solo": False,
                "bars": bars,
            }
        )
    first = song.bars[0] if song.bars else None
    t = title or song.title or "Untitled"
    a = artist if artist is not None else song.artist
    from ..theory import pc_name

    key = f"{pc_name(song.key_tonic, flats)} {song.key_mode}"
    return {
        "version": 2,
        "title": t[:300],
        "subtitle": f"Fingerstyle guitar duet · {key}"[:300],
        "composer": (f"{a} · arr. for two guitars" if a else "Arranged for two guitars")[:300],
        "tempo": float(round(first.tempo)) if first else 90,
        "meter": list(first.meter) if first else [4, 4],
        "defaultLetRing": True,
        "activeTrack": 0,
        "view": "score",
        "tracks": tracks,
        "arranger": {**arr.report, "version": _version(), "analysis_version": _analysis_version()},
    }


def write_atelier(arr: Arrangement, path: str, **kw) -> dict:
    data = to_atelier(arr, **kw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return data


def _version() -> str:
    from .. import __version__

    return __version__


def _analysis_version() -> int:
    from ..audio.pipeline import ANALYSIS_VERSION

    return ANALYSIS_VERSION
