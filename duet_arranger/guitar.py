"""Fretboard model: tunings, string tension limits and harmonic nodes.

Strings are numbered like Atelier: string 1 is the highest (thinnest) string,
string 6 the lowest. A tuning is a tuple of open-string MIDI pitches ordered
string 1..6. Frets are always counted from the capo (fret 0 = capo/nut).
"""

from __future__ import annotations

from dataclasses import dataclass

STANDARD = (64, 59, 55, 50, 45, 40)

# Named tunings (string 1..6). Open tunings get re-pitched to fit the song's
# key by ``key_shifted_tunings`` so they are not limited to these roots.
NAMED_TUNINGS: dict[str, tuple[int, ...]] = {
    "Standard": STANDARD,
    "Drop D": (64, 59, 55, 50, 45, 38),
    "Double Drop D": (62, 59, 55, 50, 45, 38),
    "DADGAD": (62, 57, 55, 50, 45, 38),
    "Open D": (62, 57, 54, 50, 45, 38),
    "Open D minor": (62, 57, 53, 50, 45, 38),
    "Open G": (62, 59, 55, 50, 43, 38),
    "Open G minor": (62, 58, 55, 50, 43, 38),
    "Open C": (64, 60, 55, 48, 43, 36),
    "Open C6": (64, 60, 55, 48, 45, 36),
    "Open E": (64, 59, 56, 52, 47, 40),
    "Open A": (64, 61, 57, 52, 45, 40),
    "DGDGCD": (62, 60, 55, 50, 43, 38),
    "CGDGBD": (62, 59, 55, 50, 43, 36),
    "Drop C": (62, 57, 53, 48, 43, 36),
    "D Standard": (62, 57, 53, 48, 43, 38),
    "Eb Standard": (63, 58, 54, 49, 44, 39),
    "Drop C#": (63, 58, 54, 49, 44, 37),
    "Open Gsus2 (DGDGAD)": (62, 57, 55, 50, 43, 38),
}

# Families whose "home key" is the given pitch class; they can be shifted so
# their open-string chord matches the song tonic.
TUNING_HOME_KEY = {
    "DADGAD": (2, "both"),
    "Open D": (2, "major"),
    "Open D minor": (2, "minor"),
    "Open G": (7, "major"),
    "Open G minor": (7, "minor"),
    "Open C": (0, "major"),
    "Open C6": (0, "major"),
    "Open E": (4, "major"),
    "Open A": (9, "major"),
    "Drop D": (2, "both"),
    "Double Drop D": (2, "both"),
    "DGDGCD": (7, "both"),
    "CGDGBD": (7, "major"),
    "Open Gsus2 (DGDGAD)": (7, "both"),
}

# Highest safe raise above standard pitch, per string (1..6), in semitones,
# and the lowest workable drop before the string flaps.
MAX_RAISE = (1, 2, 2, 2, 2, 2)
MAX_DROP = (-4, -4, -4, -5, -5, -6)

# Natural harmonic nodes: fret -> (semitones above open string, difficulty)
# The 12th and 7th fret nodes ring clearly and sustain; the 5th (like the
# 24th) is thin and short-lived, and the 4th/9th are hard to sound at all, so
# those are kept for rare cases.
NATURAL_HARMONICS: dict[float, tuple[int, float]] = {
    12: (12, 0.3),
    7: (19, 0.6),
    19: (19, 0.9),
    5: (24, 4.5),  # hard to hit cleanly and to combine with anything: rarely worth it
    4: (28, 4.0),
    9: (28, 4.0),
}


@dataclass(frozen=True)
class Tuning:
    name: str
    pitches: tuple[int, ...]  # string 1..6
    flats: bool = False  # spell note names with flats (keys like F, Bb, Eb)

    @property
    def label(self) -> str:
        from .theory import pc_name

        flats = self.flats or "Eb" in self.name or "Db" in self.name
        return " ".join(pc_name(p, flats) for p in reversed(self.pitches))

    def tension_penalty(self) -> float | None:
        """Cost of retuning from standard; None if unsafe."""
        cost = 0.0
        for i, (p, s) in enumerate(zip(self.pitches, STANDARD)):
            d = p - s
            if d > MAX_RAISE[i] or d < MAX_DROP[i]:
                return None
            if d > 0:
                cost += 0.6 * d * d
            elif d < 0:
                cost += 0.08 * d * d
        return cost

    def raises_strings(self) -> bool:
        """Does it tune any string above standard pitch? (Extra tension: the
        automatic search never asks for that; a tuning you name may.)"""
        return any(p > s for p, s in zip(self.pitches, STANDARD))

    def changed_strings(self) -> int:
        return sum(1 for p, s in zip(self.pitches, STANDARD) if p != s)


def tuning_by_name(name: str) -> Tuning:
    return Tuning(name, NAMED_TUNINGS[name])


def key_shifted_tunings(tonic: int, mode: str) -> list[Tuning]:
    """Named tunings plus open tunings re-pitched to the song's tonic."""
    from .theory import pc_name

    from .theory import prefers_flats

    flats = prefers_flats(tonic, mode)
    out: dict[tuple, Tuning] = {}
    for name, pitches in NAMED_TUNINGS.items():
        t = Tuning(name, pitches, flats)
        if t.tension_penalty() is not None and not t.raises_strings():
            out[pitches] = t
        if name in TUNING_HOME_KEY:
            home, modes = TUNING_HOME_KEY[name]
            if modes != "both" and modes != mode:
                continue
            target = tonic if mode == "major" or modes != "major" else tonic
            for shift in (target - home, target - home - 12, target - home + 12):
                if shift == 0 or not -5 <= shift <= 2:
                    continue
                shifted = tuple(p + shift for p in pitches)
                tt = Tuning(f"{name} to {pc_name(target, flats)} ({'+' if shift > 0 else ''}{shift})", shifted, flats)
                if tt.tension_penalty() is not None and not tt.raises_strings() and shifted not in out:
                    out[shifted] = tt
    return list(out.values())


@dataclass(frozen=True)
class Placement:
    """How one note is produced on the instrument.

    kind:
      fret  – left hand frets ``fret`` (>=1)
      open  – open string (fret 0)
      nh    – natural harmonic touched at node ``fret``
      ah    – artificial harmonic: LH frets ``fret`` (>=0), RH touches fret+12
      tap   – right-hand tap at ``fret``
      dead  – muted percussive pluck ("x")
    """

    string: int
    fret: float
    kind: str

    @property
    def lh_fret(self) -> float | None:
        if self.kind in ("fret",):
            return self.fret
        if self.kind == "ah" and self.fret > 0:
            return self.fret
        if self.kind == "nh":
            return self.fret
        return None


def sounding_pitch(p: Placement, tuning: tuple[int, ...], capo: int) -> int | None:
    base = tuning[p.string - 1] + capo
    if p.kind in ("fret", "open", "tap"):
        return base + int(p.fret)
    if p.kind == "nh":
        return base + NATURAL_HARMONICS[p.fret][0]
    if p.kind == "ah":
        return base + int(p.fret) + 12
    return None


def comfortable_span(position: float) -> int:
    """Fret span (max-min) the hand covers without stretching at a position."""
    if position >= 10:
        return 5
    if position >= 5:
        return 4
    return 3


def max_span(position: float) -> int:
    return comfortable_span(position) + 1
