"""MIDI preview of the arrangement (two guitar tracks, panned apart)."""

from __future__ import annotations

import mido

from ..arrange.duet import Arrangement
from ..model import TPQ

PPQ = 480


def write_midi(arr: Arrangement, path: str) -> None:
    song = arr.song
    mid = mido.MidiFile(ticks_per_beat=PPQ)
    scale = PPQ // TPQ
    conductor = mido.MidiTrack()
    mid.tracks.append(conductor)
    events = []
    last_tempo = None
    last_meter = None
    for b in song.bars:
        if b.tempo != last_tempo:
            events.append((b.start * scale, 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(b.tempo), time=0)))
            last_tempo = b.tempo
        if b.meter != last_meter:
            events.append((b.start * scale, 0, mido.MetaMessage("time_signature", numerator=b.beats, denominator=b.unit, time=0)))
            last_meter = b.meter
    _flush(conductor, events)
    for ch, g in enumerate(arr.guitars):
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=f"{g.name} ({g.tuning.name}{', capo ' + str(g.capo) if g.capo else ''})", time=0))
        tr.append(mido.Message("program_change", channel=ch, program=25, time=0))
        tr.append(mido.Message("control_change", channel=ch, control=10, value=40 if ch == 0 else 88, time=0))
        evs = []
        for n in g.placed:
            if n.pitch is None or n.fret == "x":
                continue
            vel = int(max(20, min(120, 40 + 80 * n.velocity)))
            evs.append((n.start * scale, 1, mido.Message("note_on", channel=ch, note=int(n.pitch), velocity=vel, time=0)))
            evs.append((max(n.end, n.start + 1) * scale, 0, mido.Message("note_off", channel=ch, note=int(n.pitch), velocity=0, time=0)))
        _flush(tr, evs)
    mid.save(path)


def _flush(track, events):
    events.sort(key=lambda e: (e[0], e[1]))
    now = 0
    for t, _, msg in events:
        msg.time = int(t - now)
        now = t
        track.append(msg)
