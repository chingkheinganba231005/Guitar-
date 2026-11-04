"""Read a multi-track MIDI file into a Song (an alternative to audio input)."""

from __future__ import annotations

import re
from statistics import mean

import mido

from .analysis import finish_analysis
from .model import TPQ, Bar, Hit, Note, Song
from .quantize import detect_triplet_beats, snap_notes

KICKS = {35, 36}
SNARES = {37, 38, 39, 40}
HATS = {42, 44, 46}


def _role_from_name(name: str, program: int | None, channel: int) -> str | None:
    n = name.lower()
    if channel == 9:
        return "drums"
    if re.search(r"vocal|voice|vox|melody|sing|lead vocal|tune", n):
        return "vocals"
    if "bass" in n or (program is not None and 32 <= program <= 39):
        return "bass"
    if re.search(r"piano|keys|rhodes|organ|epiano", n) or (program is not None and program <= 7) or (program is not None and 16 <= program <= 23):
        return "piano"
    if "guitar" in n or (program is not None and 24 <= program <= 31):
        return "guitar"
    if re.search(r"string|violin|cello|viola|pad", n) or (program is not None and 40 <= program <= 55) or (program is not None and 88 <= program <= 95):
        return "strings"
    if re.search(r"synth|lead", n) or (program is not None and 80 <= program <= 87):
        return "synth"
    return None


def read_midi(path: str, title: str | None = None) -> Song:
    mid = mido.MidiFile(path)
    ppq = mid.ticks_per_beat
    scale = TPQ / ppq
    tempos: list[tuple[int, float]] = []
    meters: list[tuple[int, int, int]] = []
    tracks: list[dict] = []
    for ti, tr in enumerate(mid.tracks):
        t = 0
        name = tr.name or f"Track {ti + 1}"
        program: dict[int, int] = {}
        on: dict[tuple[int, int], tuple[int, int]] = {}
        notes: dict[int, list] = {}
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                tempos.append((t, mido.tempo2bpm(msg.tempo)))
            elif msg.type == "time_signature":
                meters.append((t, msg.numerator, msg.denominator))
            elif msg.type == "track_name":
                name = msg.name
            elif msg.type == "program_change":
                program[msg.channel] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                on[(msg.channel, msg.note)] = (t, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                k = (msg.channel, msg.note)
                if k in on:
                    st, vel = on.pop(k)
                    notes.setdefault(msg.channel, []).append((st, t, msg.note, vel))
        for ch, ns in notes.items():
            tracks.append({"name": name, "channel": ch, "program": program.get(ch), "notes": ns})
    end_midi = max((n[1] for tr in tracks for n in tr["notes"]), default=ppq * 4)
    tempos.sort()
    meters.sort()
    if not tempos or tempos[0][0] > 0:
        tempos.insert(0, (0, 120.0))
    if not meters or meters[0][0] > 0:
        meters.insert(0, (0, 4, 4))
    song = Song(title=title or path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    # bars from the time-signature map
    t = 0
    mi = 0
    while t < end_midi:
        while mi + 1 < len(meters) and meters[mi + 1][0] <= t:
            mi += 1
        num, den = meters[mi][1], meters[mi][2]
        tempo = [bpm for tt, bpm in tempos if tt <= t][-1]
        compound = den == 8 and num % 3 == 0 and num > 3
        bar = Bar(int(round(t * scale)), num, den, float(tempo), int(TPQ * 1.5) if compound else TPQ)
        song.bars.append(bar)
        t += int(num * ppq * 4 / den)
    onsets = []
    parts: dict[str, list[Note]] = {}
    hits: list[Hit] = []
    unassigned = []
    for tr in tracks:
        role = _role_from_name(tr["name"], tr["program"], tr["channel"])
        if role == "drums":
            for st, en, p, vel in tr["notes"]:
                kind = "kick" if p in KICKS else "snare" if p in SNARES else "hat" if p in HATS else None
                if kind:
                    hits.append(Hit(int(round(st * scale)), kind, vel / 127))
            continue
        ns = [Note(st * scale, max(en, st + 1) * scale, p, vel / 127, source=role or "other") for st, en, p, vel in tr["notes"]]
        onsets += [n.start for n in ns]
        if role is None:
            unassigned.append((tr, ns))
        else:
            parts.setdefault(role, []).extend(ns)
    # Unnamed tracks: highest monophonic-ish line is the melody, lowest the bass.
    if unassigned:
        def poly(ns):
            starts = [n.start for n in ns]
            return len(starts) / max(1, len(set(starts)))

        if "vocals" not in parts:
            cands = [x for x in unassigned if poly(x[1]) < 1.3 and len(x[1]) >= 8]
            if cands:
                best = max(cands, key=lambda x: mean(n.pitch for n in x[1]))
                unassigned.remove(best)
                parts["vocals"] = best[1]
        if "bass" not in parts:
            cands = [x for x in unassigned if poly(x[1]) < 1.3 and mean(n.pitch for n in x[1]) < 52]
            if cands:
                best = min(cands, key=lambda x: mean(n.pitch for n in x[1]))
                unassigned.remove(best)
                parts["bass"] = best[1]
        for tr, ns in unassigned:
            parts.setdefault("other", []).extend(ns)
    song.triplet_beats = detect_triplet_beats(song, onsets)
    for k in list(parts):
        parts[k] = snap_notes(parts[k], song)
    song.parts = parts
    song.melody = list(parts.get("vocals", []))
    song.bass = list(parts.get("bass", []))
    song.hits = sorted(hits, key=lambda h: h.start)
    from .quantize import Grid

    g = Grid(song)
    for h in song.hits:
        h.start = g.snap(h.start)
    return finish_analysis(song)
