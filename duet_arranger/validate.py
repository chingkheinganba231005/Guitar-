"""Independent playability audit of an exported Atelier project.

This reads the written project (not the solver's internal state), rebuilds
what each hand must do at every beat — including notes held through ties —
and reports anything a human could not play.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fingering import SolverConfig, lh_eval
from .guitar import NATURAL_HARMONICS

DEFAULT_MAX_FRET = 20


@dataclass
class Issue:
    track: str
    bar: int
    event: int
    kind: str
    detail: str

    def __str__(self):
        return f"{self.track} bar {self.bar} beat {self.event}: {self.kind} – {self.detail}"


def _beats(e) -> float:
    dots = 1.75 if e["dots"] == 2 else (1.5 if e["dots"] else 1.0)
    return 4 / e["den"] * dots * e["ratio"][1] / e["ratio"][0]


def audit_track(track: dict, project: dict, max_fret: int = DEFAULT_MAX_FRET) -> list[Issue]:
    issues: list[Issue] = []
    capo = track.get("capo", 0)
    cfg = SolverConfig(max_fret=max_fret - capo)
    tempo = project.get("tempo", 90)
    sec = 0.0
    prev_interval = None
    prev_time = -10.0
    name = track.get("name", "track")
    for bi, bar in enumerate(track["bars"]):
        if bar.get("tempo"):
            tempo = bar["tempo"]
        for ei, e in enumerate(bar["events"]):
            where = (bi + 1, ei + 1)
            notes = e["notes"]
            strings = [n["string"] for n in notes]
            if len(strings) != len(set(strings)):
                issues.append(Issue(name, *where, "string", "two notes on one string"))
            attacks = [n for n in notes if not n.get("tie")]
            holds = [n for n in notes if n.get("tie")]
            points = []
            played = {}
            for n in notes:
                f = n["fret"]
                if f == "x":
                    played[n["string"]] = 0
                    continue
                h = n.get("harmonic", "")
                tech = n.get("tech", [])
                if h == "natural":
                    if f not in NATURAL_HARMONICS:
                        issues.append(Issue(name, *where, "harmonic", f"no natural harmonic node at fret {f}"))
                    if not n.get("tie"):
                        points.append((n["string"], f, "touch"))
                    played[n["string"]] = -1
                    continue
                if f + capo > (24 if h else max_fret) and "RH" not in tech:
                    issues.append(Issue(name, *where, "reach", f"fret {f} beyond the fretboard"))
                if h == "artificial":
                    interval = n.get("harmonicInterval", 12)
                    touch = f + {12: 12, 19: 7, 24: 5}.get(interval, 12)
                    if touch + capo > 24:
                        issues.append(Issue(name, *where, "harmonic", "artificial harmonic touch point off the fretboard"))
                if "RH" in tech:
                    played[n["string"]] = f
                    continue
                played[n["string"]] = f
                if f > 0:
                    points.append((n["string"], f, "press"))
            lh = lh_eval(points, played, cfg) if points else (0.0, None, False)
            if lh is None:
                frets = sorted({p[1] for p in points})
                issues.append(Issue(name, *where, "left hand", f"cannot fret {len(points)} points at frets {frets}"))
                interval = None
            else:
                interval = lh[1]
            # right hand
            taps = [n for n in attacks if "RH" in n.get("tech", [])]
            lh_taps = [n for n in attacks if "LH" in n.get("tech", [])]
            ahs = [n for n in attacks if n.get("harmonic") == "artificial"]
            plucked = [n for n in attacks if n not in taps and n not in lh_taps and n not in ahs]
            dead_groups = 0
            pitched_plucked = [n for n in plucked if n["fret"] != "x"]
            if any(n["fret"] == "x" for n in plucked):
                dead_groups = 1
            n_rh = len(pitched_plucked) + dead_groups
            if len(taps) > 1:
                issues.append(Issue(name, *where, "right hand", "more than one right-hand tap at once"))
            if taps and (pitched_plucked or dead_groups or ahs):
                issues.append(Issue(name, *where, "right hand", "plucking while the right hand taps"))
            if ahs:
                if len(ahs) > 1 or len(pitched_plucked) > 1:
                    issues.append(Issue(name, *where, "right hand", "artificial harmonic with too many other plucked notes"))
                elif pitched_plucked and pitched_plucked[0]["string"] <= ahs[0]["string"]:
                    issues.append(Issue(name, *where, "right hand", "thumb note above an artificial harmonic"))
            if n_rh > 4 and not e.get("strum"):
                low = sorted((n["string"] for n in pitched_plucked), reverse=True)
                if not (n_rh == 5 and len(low) >= 2 and low[0] - low[1] == 1 and low[1] >= 4):
                    issues.append(Issue(name, *where, "right hand", f"{n_rh} simultaneous plucks without a strum"))
            # movement between beats
            if interval is not None:
                if prev_interval is not None:
                    lo = max(prev_interval[0], interval[0])
                    hi = min(prev_interval[1], interval[1])
                    d = 0 if lo <= hi else lo - hi
                    dt = sec - prev_time
                    if d >= 5 and dt < 0.09:
                        issues.append(Issue(name, *where, "shift", f"{d}-fret jump in {dt * 1000:.0f} ms"))
                prev_interval = interval
                prev_time = sec
            sec += _beats(e) * 60.0 / tempo
    return issues


def audit_project(project: dict, max_fret: int = DEFAULT_MAX_FRET) -> list[Issue]:
    tracks = project.get("tracks")
    if not tracks:
        tracks = [{"name": "Guitar", "bars": project["bars"], "capo": project.get("capo", 0), "tuning": project["tuning"]}]
    out = []
    for t in tracks:
        out += audit_track(t, project, max_fret)
    return out


def bar_fill_errors(project: dict) -> list[str]:
    """Every bar must be rhythmically complete (ties across barlines depend on it)."""
    errs = []
    meter = project.get("meter", [4, 4])
    for t in project.get("tracks", []):
        m = meter
        for bi, b in enumerate(t["bars"]):
            if b.get("meter"):
                m = b["meter"]
            cap = m[0] * 4 / m[1]
            used = sum(_beats(e) for e in b["events"])
            if abs(used - cap) > 1e-6:
                errs.append(f"{t['name']} bar {bi + 1}: {used:.3f} of {cap} beats")
    return errs
