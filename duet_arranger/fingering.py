"""Physically-aware fingering search.

Every onset in a guitar part becomes a *slice*: the notes attacked at that
instant plus the notes still sounding from earlier attacks (holds). For each
slice we enumerate realisations — which string/fret/technique produces every
note — that a real pair of hands can make:

* left hand: at most four fingers (+ thumb over the neck on the 6th string),
  barres that never cover a string that must ring open, fret spans that depend
  on the position on the neck, held notes keep their finger down;
* right hand: at most four independent plucking fingers (five with a thumb
  brush, more only as a strum), one right-hand tap at a time with no other
  plucks, artificial harmonics with at most one thumb note beneath them;
* harmonics only on real nodes of the chosen tuning.

A beam search (Viterbi over dynamically generated states) then picks the
sequence of realisations with the lowest total cost, where moving the hand
quickly over large distances, cutting sustaining notes short, dropping notes
or displacing them by an octave all cost something.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Optional

from .guitar import NATURAL_HARMONICS, Placement, comfortable_span, max_span
from .model import TPQ, Hit, Note

INF = float("inf")


@dataclass
class SolverConfig:
    max_fret: int = 20  # highest usable fret on an acoustic with a cutaway
    beam: int = 28
    per_key: int = 18
    allow_taps: bool = True
    allow_harmonics: bool = True
    allow_thumb: bool = True
    percussive: bool = False


class Option:
    __slots__ = ("places", "cost", "pitch", "shift", "techs", "attach", "label")

    def __init__(self, places, cost, pitch=None, shift=0, techs=(), attach=None, label=""):
        self.places: tuple = places  # tuple[Placement, ...]; () = dropped / attach-only
        self.cost: float = cost
        self.pitch = pitch
        self.shift = shift
        self.techs = techs
        self.attach = attach  # None | "any" | "bass" (percussion carried by another note)
        self.label = label

    @property
    def dropped(self) -> bool:
        return not self.places and self.attach is None

    @property
    def place(self) -> Optional[Placement]:
        return self.places[0] if self.places else None

    def __repr__(self):  # pragma: no cover - debugging helper
        return f"Option({self.places}, {self.cost:.2f}, shift={self.shift}, techs={self.techs}, attach={self.attach})"


class Item:
    """A note or percussion hit to realise."""

    __slots__ = ("id", "start", "end", "pitch", "role", "priority", "tags", "kind", "options", "note")

    def __init__(self, note: Note | None = None, hit: Hit | None = None):
        if note is not None:
            self.id = note.id
            self.start, self.end = note.start, note.end
            self.pitch = note.pitch
            self.role = note.role
            self.priority = note.priority
            self.tags = note.tags
            self.kind = "note"
            self.note = note
        else:
            self.id = hit.id
            self.start = self.end = hit.start
            self.pitch = None
            self.role = "perc"
            self.priority = 1.5
            self.tags = {hit.kind}
            self.kind = hit.kind
            self.note = hit
        self.options: list[Option] = []


# ---------------------------------------------------------------- options


def _melodic_role(role: str) -> bool:
    return role in ("melody", "fill")


def note_options(item: Item, tuning, capo: int, cfg: SolverConfig, beat_ticks: int = TPQ) -> list[Option]:
    """All ways to produce this note, each with a static cost."""
    opts: list[Option] = []
    role = item.role
    tags = item.tags
    harmonic_only = "harmonic_only" in tags
    prefer_h = "prefer_harmonic" in tags or harmonic_only
    long_note = (item.end - item.start) >= int(1.5 * beat_ticks)
    reach = cfg.max_fret - capo
    shifts = [0]
    if role in ("comp", "harmony", "support", "harmonic", "fill") or ("octave_ok" in tags):
        shifts += [12, -12]
    elif role == "bass":
        shifts += [12]
    for shift in shifts:
        p = item.pitch + shift
        shift_cost = 0.0 if shift == 0 else (0.6 + 0.4 * item.priority if role != "bass" else 3.0 + 0.2 * item.priority)
        for s in range(1, 7):
            base = tuning[s - 1] + capo
            f = p - base
            if not harmonic_only and 0 <= f <= reach:
                kind = "open" if f == 0 else "fret"
                c = 0.1
                if kind == "fret":
                    c += 0.035 * f + 0.12 * max(0, f - 12)
                if _melodic_role(role):
                    c += {6: 1.4, 5: 1.0, 4: 0.35}.get(s, 0.0)
                    if kind == "open":
                        c += 0.25 + (0.35 if long_note else 0.0)
                elif role == "bass":
                    c += {1: 3.0, 2: 2.2, 3: 0.9}.get(s, 0.0)
                    c += 0.1 * max(0, f - 7)
                    if kind == "open":
                        c -= 0.25
                else:  # inner harmony / comp
                    if kind == "open":
                        c -= 0.2
                    c += 0.08 * max(0, f - 9)
                    if s == 6 and role != "bass":
                        c += 0.4
                opts.append(Option((Placement(s, f, kind),), c + shift_cost, p, shift))
                if (
                    cfg.allow_taps
                    and _melodic_role(role)
                    and kind == "fret"
                    and f >= 7
                    and shift == 0
                ):
                    opts.append(Option((Placement(s, f, "tap"),), c + 2.6, p, shift, ("RH",)))
            if cfg.allow_harmonics:
                for node, (semi, diff) in NATURAL_HARMONICS.items():
                    if base + semi == p and node + capo <= cfg.max_fret:
                        c = diff + (-1.2 if prefer_h else (3.2 if _melodic_role(role) else 4.5))
                        if harmonic_only:
                            c = diff * 0.6
                        opts.append(Option((Placement(s, node, "nh"),), c + shift_cost, p, shift))
                fa = p - 12 - base
                if 0 <= fa <= 12 and fa + 12 + capo <= 24 and (prefer_h or harmonic_only):
                    c = 1.9 + (0.3 if fa > 0 else 0.0) - (0.6 if prefer_h else 0.0)
                    opts.append(Option((Placement(s, fa, "ah"),), c + shift_cost, p, shift))
    if role == "melody" and not any(o.shift == 0 for o in opts):
        # Out of range: an octave displacement beats losing a melody note.
        for shift in (12, -12, 24, -24):
            p = item.pitch + shift
            for s in range(1, 7):
                f = p - tuning[s - 1] - capo
                if 0 <= f <= reach:
                    kind = "open" if f == 0 else "fret"
                    opts.append(Option((Placement(s, f, kind),), 9.0 + abs(shift) / 6 + 0.04 * f, p, shift))
            if any(o.shift == shift for o in opts):
                break
    if role == "melody":
        drop = 120.0 * item.priority / 100.0 + 40.0
    else:
        drop = item.priority * 1.6 + 0.4
    opts.append(Option((), drop, None, 0))
    opts.sort(key=lambda o: o.cost)
    # Keep the list short: the best placements per string are enough.
    seen = set()
    pruned = []
    for o in opts:
        key = (o.place.string, o.place.kind) if o.places else ("drop",)
        if key in seen:
            continue
        seen.add(key)
        pruned.append(o)
    return pruned[:10]


def hit_options(item: Item, cfg: SolverConfig) -> list[Option]:
    opts: list[Option] = []
    if item.kind == "kick":
        opts.append(Option((), 0.12, techs=("BW",), attach="any", label="kick"))
        for s in (6, 5):
            opts.append(Option((Placement(s, 0, "dead"),), 0.45 + (6 - s) * 0.1, techs=("BW",), label="kick"))
        drop = 1.4
    elif item.kind == "snare":
        opts.append(Option((), 0.1, techs=("SB",), attach="bass", label="snare"))
        opts.append(Option((), 0.35, techs=("WT",), attach="chord", label="snare"))
        for group in ((3, 2), (4, 3), (5, 4), (2, 1)):
            opts.append(
                Option(tuple(Placement(s, 0, "dead") for s in group), 0.4 + 0.05 * group[0], techs=("WT",), label="snare")
            )
        drop = 1.8
    else:
        drop = 0.2
    opts.append(Option((), drop))
    return opts


# ------------------------------------------------------------ hand checks


def lh_eval(points, played: dict[int, float], cfg: SolverConfig):
    """Left-hand feasibility of simultaneous fretted/touched points.

    ``points``: list of (string, fret, mode) with mode "press" or "touch".
    ``played``: string -> fret for every string sounding in the slice (0 = open).
    Returns (cost, (lo, hi) index-finger interval, thumb_used) or None.
    """
    if not points:
        return 0.0, None, False
    frets = [p[1] for p in points]
    lo = min(frets)
    hi = max(frets)
    if hi > cfg.max_fret:
        return None
    span = hi - lo
    if span > max_span(lo):
        return None
    cost = 0.0
    comf = comfortable_span(lo)
    if span > comf:
        cost += 2.4 * (span - comf)
    groups: dict[tuple, list[int]] = {}
    for s, f, mode in points:
        groups.setdefault((f, mode), []).append(s)
    fingers = 0
    for (f, mode), strs in groups.items():
        if len(strs) == 1:
            fingers += 1
            continue
        smin, smax = min(strs), max(strs)
        blocked = False
        for s2 in range(smin, smax + 1):
            v = played.get(s2)
            if v is not None and (v == 0 or v < f):
                blocked = True
                break
        if mode == "touch":
            if not blocked:
                fingers += 1
                cost += 0.2
            else:
                fingers += len(strs)
            continue
        if not blocked and f == lo:
            fingers += 1
            cost += 0.7 if smax - smin >= 3 else 0.35
        elif not blocked and smax - smin <= 2:
            fingers += 1
            cost += 1.0
        else:
            fingers += len(strs)
            if len(strs) > 1:
                cost += 0.15 * (len(strs) - 1)
    # Finger crossing: several separate fingers at the lowest fret on thin strings
    # while a higher-fret finger has to reach back over them to thicker strings.
    low_sep = [s for (f, mode), strs in groups.items() if f == lo and len(strs) >= 2 for s in strs]
    if len(low_sep) >= 2 and fingers >= 4:
        higher = [s for s, f, mode in points if f > lo]
        if any(min(low_sep) < s < max(low_sep) for s in higher):
            cost += 1.6
    thumb = False
    if fingers > 4:
        if not cfg.allow_thumb:
            return None
        ok = any(s == 6 and mode == "press" and f <= lo + 1 for s, f, mode in points) and lo <= 7
        if ok and fingers - 1 <= 4:
            fingers -= 1
            thumb = True
            cost += 2.2
        else:
            return None
    cost += 0.12 * fingers
    reach = comf if span <= comf else max_span(lo)
    interval = (max(1, hi - reach), lo)
    return cost, interval, thumb


def rh_eval(attacks: list[tuple[Item, Option]], hold_tap: bool):
    """Right-hand feasibility of the notes attacked together.

    Returns (cost, flags) or None. flags: dict with strum / lh_tap ids.
    """
    plucked = []
    taps = []
    ahs = []
    dead_groups = 0
    for it, o in attacks:
        if not o.places:
            continue
        k = o.places[0].kind
        if k == "tap":
            taps.append((it, o))
        elif k == "ah":
            ahs.append((it, o))
        elif k == "dead":
            dead_groups += 1
        else:
            plucked.append((it, o))
    flags: dict = {}
    cost = 0.0
    if hold_tap and (plucked or ahs or dead_groups):
        return None
    if taps:
        if len(taps) > 1 or ahs or dead_groups:
            return None
        if len(plucked) > 1:
            return None
        if plucked:
            it, o = plucked[0]
            if o.places[0].kind != "fret":
                return None
            flags["lh_tap"] = it.id
            cost += 2.2
        return cost, flags
    if ahs:
        if len(ahs) > 1 or dead_groups:
            return None
        if len(plucked) > 1:
            return None
        if plucked and plucked[0][1].places[0].string <= ahs[0][1].places[0].string:
            return None
        return 0.1, flags
    n = len(plucked) + dead_groups
    if n <= 4:
        cost += 0.04 * n
    else:
        strings = sorted((o.places[0].string for _, o in plucked), reverse=True)
        if n == 5 and len(strings) >= 2 and strings[0] - strings[1] == 1 and strings[1] >= 4:
            cost += 0.9  # thumb brushes the two lowest strings
        else:
            cost += 1.5
            flags["strum"] = True
    return cost, flags


# ------------------------------------------------------------------ slices


@dataclass
class Slice:
    tick: int
    sec: float
    attacks: list[Item]
    holds: list[Item]
    next_tick: int = 0


def build_slices(items: list[Item], seconds_at: Callable[[float], float]) -> list[Slice]:
    starts = sorted({it.start for it in items})
    by_start: dict[int, list[Item]] = {}
    for it in items:
        by_start.setdefault(it.start, []).append(it)
    pitched = sorted((it for it in items if it.kind == "note"), key=lambda it: it.start)
    slices = []
    active: list[Item] = []
    pi = 0
    for idx, t in enumerate(starts):
        while pi < len(pitched) and pitched[pi].start < t:
            active.append(pitched[pi])
            pi += 1
        active = [a for a in active if a.end > t]
        attacks = sorted(by_start[t], key=lambda it: -it.priority)
        slices.append(Slice(t, seconds_at(t), attacks, list(active)))
    for i, s in enumerate(slices):
        s.next_tick = slices[i + 1].tick if i + 1 < len(slices) else s.tick + TPQ * 4
    return slices


# ------------------------------------------------------------------ search


class Cand:
    __slots__ = ("assign", "dropped_holds", "cost", "lh", "thumb", "flags", "strings", "mel_string", "has_tap", "plucks")

    def __init__(self):
        self.assign: dict[int, Option] = {}
        self.dropped_holds: list[int] = []
        self.cost = 0.0
        self.lh = None
        self.thumb = False
        self.flags: dict = {}
        self.strings: tuple = ()
        self.mel_string = None
        self.has_tap = False
        self.plucks = False


class State:
    __slots__ = ("cost", "assign", "lh", "lh_time", "last_attack", "mel_string", "parent", "cand", "tap_time", "pluck_time")

    def __init__(self):
        self.cost = 0.0
        self.assign: dict[int, Option] = {}
        self.lh = None
        self.lh_time = -10.0
        self.last_attack: dict[int, float] = {}
        self.mel_string = None
        self.parent: Optional["State"] = None
        self.cand: Optional[Cand] = None
        self.tap_time = -10.0
        self.pluck_time = -10.0


def _interval_distance(a, b) -> float:
    if a is None or b is None:
        return 0.0
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo <= hi:
        return 0.0
    return lo - hi


def _intersect(a, b):
    if a is None:
        return b
    if b is None:
        return a
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return (lo, hi) if lo <= hi else b


class Solver:
    def __init__(self, tuning, capo: int, cfg: SolverConfig, seconds_at: Callable[[float], float], beat_ticks: int = TPQ):
        self.tuning = tuning
        self.capo = capo
        self.cfg = cfg
        self.seconds_at = seconds_at
        self.beat_ticks = beat_ticks
        self._cache: dict = {}

    # -- candidate enumeration for one slice given the kept holds -------------
    def candidates(self, si: int, sl: Slice, kept: tuple) -> list[Cand]:
        key = (si, kept)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        K = self.cfg.per_key
        best: list = []  # max-heap via negative cost: (-cost, counter, cand)
        counter = [0]
        attacks = sl.attacks
        hold_items = {it.id: it for it in sl.holds}
        kept_list = list(kept)  # ((id, Option), ...)

        used_strings: dict[int, float] = {}
        points: list = []
        chosen: list[tuple[Item, Option]] = []
        dropped: list[int] = []

        def bound() -> float:
            return -best[0][0] if len(best) >= K else INF

        def add_points(o: Option):
            added = []
            for pl in o.places:
                lf = pl.lh_fret
                if lf is not None and lf > 0:
                    pt = (pl.string, lf, "touch" if pl.kind == "nh" else "press")
                    points.append(pt)
                    added.append(pt)
            return added

        def partial_ok() -> bool:
            if not points:
                return True
            lo = min(p[1] for p in points)
            hi = max(p[1] for p in points)
            return hi - lo <= max_span(lo) and hi <= self.cfg.max_fret

        hold_tap = False
        base_cost = 0.0
        # Holds are fixed by the state; drops of holds were decided by the state as well
        # except when the string is needed: we allow dropping a hold here too.
        hold_choices = []
        for hid, o in kept_list:
            it = hold_items.get(hid)
            if it is None:
                continue
            remaining = max(0, it.end - sl.tick)
            weight = {"melody": 0.6, "bass": 0.5, "comp": 0.35, "support": 0.4, "inner": 0.35, "harmonic": 0.3}.get(it.role, 0.5)
            drop_cost = it.priority * min(1.0, remaining / (2 * TPQ)) * weight + 0.05
            hold_choices.append((it, o, drop_cost))

        def finish(acc_cost: float):
            # attach-type percussion needs a carrier
            pitched_attacks = [(it, o) for it, o in chosen if it.kind == "note" and o.places]
            extra = 0.0
            for it, o in chosen:
                if o.attach == "any" and not pitched_attacks:
                    return
                if o.attach == "bass":
                    if not any(i2.role == "bass" and o2.places[0].string >= 4 and o2.places[0].kind in ("fret", "open") for i2, o2 in pitched_attacks):
                        return
                if o.attach == "chord":
                    if sum(1 for i2, o2 in pitched_attacks if o2.places[0].kind in ("fret", "open")) < 2:
                        return
            played = dict(used_strings)
            lh = lh_eval(points, played, self.cfg)
            if lh is None:
                return
            lh_cost, interval, thumb = lh
            rh = rh_eval([(it, o) for it, o in chosen], hold_tap)
            if rh is None:
                return
            rh_cost, flags = rh
            total = acc_cost + lh_cost + rh_cost + extra
            if total >= bound():
                return
            c = Cand()
            c.assign = {it.id: o for it, o in chosen}
            for hid, o in kept_list:
                if hid in hold_items and hid not in dropped:
                    c.assign[hid] = o
            c.dropped_holds = list(dropped)
            c.cost = total
            c.lh = interval
            c.thumb = thumb
            c.flags = flags
            c.strings = tuple(sorted(o.places[0].string for it, o in chosen if o.places and o.places[0].kind != "tap"))
            mel = [o.places[0].string for it, o in chosen if it.role == "melody" and o.places]
            c.mel_string = mel[0] if mel else None
            c.has_tap = any(o.places and o.places[0].kind == "tap" for it, o in chosen)
            c.plucks = any(o.places and o.places[0].kind != "tap" for it, o in chosen)
            counter[0] += 1
            if len(best) < K:
                heapq.heappush(best, (-total, counter[0], c))
            else:
                heapq.heapreplace(best, (-total, counter[0], c))

        def dfs_attacks(i: int, acc: float):
            if acc >= bound():
                return
            if i == len(attacks):
                finish(acc)
                return
            it = attacks[i]
            for o in it.options:
                if acc + o.cost >= bound():
                    continue
                if any(pl.string in used_strings for pl in o.places):
                    continue
                for pl in o.places:
                    used_strings[pl.string] = pl.fret if pl.kind in ("fret", "ah") else (0 if pl.kind in ("open", "dead") else (-1 if pl.kind == "nh" else pl.fret))
                added = add_points(o)
                if partial_ok():
                    chosen.append((it, o))
                    dfs_attacks(i + 1, acc + o.cost)
                    chosen.pop()
                for pt in added:
                    points.remove(pt)
                for pl in o.places:
                    del used_strings[pl.string]

        def dfs_holds(i: int, acc: float):
            nonlocal hold_tap
            if i == len(hold_choices):
                dfs_attacks(0, acc)
                return
            it, o, drop_cost = hold_choices[i]
            pl = o.places[0]
            # keep
            if pl.string not in used_strings:
                used_strings[pl.string] = pl.fret if pl.kind in ("fret", "ah") else (0 if pl.kind == "open" else -1)
                added = []
                if pl.kind in ("fret", "ah") and pl.fret > 0:
                    pt = (pl.string, pl.fret, "press")
                    points.append(pt)
                    added.append(pt)
                was_tap = hold_tap
                if pl.kind == "tap":
                    hold_tap = True
                if partial_ok():
                    dfs_holds(i + 1, acc)
                hold_tap = was_tap
                for pt in added:
                    points.remove(pt)
                del used_strings[pl.string]
            # drop
            dropped.append(it.id)
            dfs_holds(i + 1, acc + drop_cost)
            dropped.pop()

        dfs_holds(0, base_cost)
        out = [c for _, _, c in sorted(best, key=lambda x: -x[0])]
        self._cache[key] = out
        return out

    # -- transition --------------------------------------------------------------
    def transition(self, st: State, c: Cand, sl: Slice) -> float:
        cost = 0.0
        t = sl.sec
        if c.lh is not None and st.lh is not None:
            d = _interval_distance(st.lh, c.lh)
            if d > 0:
                dt = max(0.03, t - st.lh_time)
                cost += d * (0.28 + 0.07 / dt)
                if d >= 5 and dt < 0.16:
                    cost += 3.0
                if d >= 9 and dt < 0.3:
                    cost += 4.0
        for s in c.strings:
            last = st.last_attack.get(s)
            if last is not None and t - last < 0.065:
                cost += 2.5
        if c.has_tap and t - st.pluck_time < 0.11:
            cost += 1.8
        if c.plucks and t - st.tap_time < 0.11:
            cost += 1.8
        if c.mel_string is not None and st.mel_string is not None:
            if c.mel_string == st.mel_string:
                cost -= 0.12
            elif abs(c.mel_string - st.mel_string) > 1:
                cost += 0.3 * (abs(c.mel_string - st.mel_string) - 1)
        return cost

    def advance(self, st: State, c: Cand, sl: Slice) -> State:
        ns = State()
        ns.cost = st.cost + c.cost + self.transition(st, c, sl)
        ns.assign = c.assign
        if c.lh is not None:
            ns.lh = _intersect(st.lh, c.lh) if _interval_distance(st.lh, c.lh) == 0 else c.lh
            ns.lh_time = sl.sec
        else:
            ns.lh = st.lh
            ns.lh_time = st.lh_time
        la = dict(st.last_attack)
        for s in c.strings:
            la[s] = sl.sec
        ns.last_attack = la
        ns.mel_string = c.mel_string if c.mel_string is not None else st.mel_string
        ns.tap_time = sl.sec if c.has_tap else st.tap_time
        ns.pluck_time = sl.sec if c.plucks else st.pluck_time
        ns.parent = st
        ns.cand = c
        return ns

    def signature(self, st: State, sl: Slice):
        future = tuple(sorted((nid, o.places) for nid, o in st.assign.items() if o.places and self._ends.get(nid, 0) > sl.next_tick))
        lh = None if st.lh is None else (round(st.lh[0]), round(st.lh[1]))
        return (future, lh, st.mel_string)

    # -- main --------------------------------------------------------------------
    def solve(self, notes: list[Note], hits: list[Hit] = ()) -> "Solution":
        items = [Item(note=n) for n in notes]
        if self.cfg.percussive:
            items += [Item(hit=h) for h in hits]
        for it in items:
            it.options = note_options(it, self.tuning, self.capo, self.cfg, self.beat_ticks) if it.kind == "note" else hit_options(it, self.cfg)
        self._ends = {it.id: it.end for it in items}
        slices = build_slices(items, self.seconds_at)
        beam = [State()]
        B = self.cfg.beam
        for si, sl in enumerate(slices):
            hold_ids = {it.id for it in sl.holds}
            groups: dict = {}
            for st in beam:
                kept = tuple(sorted(((nid, o) for nid, o in st.assign.items() if nid in hold_ids and o.places), key=lambda x: x[0]))
                key = tuple((nid, o.places) for nid, o in kept)
                groups.setdefault(key, []).append((st, kept))
            nxt: dict = {}
            for key, members in groups.items():
                cands = self.candidates(si, sl, members[0][1])
                for st, kept in members:
                    for c in cands:
                        ns = self.advance(st, c, sl)
                        sig = self.signature(ns, sl)
                        old = nxt.get(sig)
                        if old is None or ns.cost < old.cost:
                            nxt[sig] = ns
            if not nxt:
                raise RuntimeError(f"No playable realisation at tick {sl.tick}")
            beam = heapq.nsmallest(B, nxt.values(), key=lambda s: s.cost)
            # free cache memory for passed slices
            if si % 64 == 0:
                self._cache = {k: v for k, v in self._cache.items() if k[0] >= si}
        best = min(beam, key=lambda s: s.cost)
        return Solution.from_path(best, slices, items, self)


@dataclass
class Realised:
    item: Item
    option: Option
    start: int
    end: int
    slice_index: int
    lh_finger: str = ""
    rh_finger: str = ""
    lh_tap: bool = False
    strum: bool = False


@dataclass
class Solution:
    cost: float
    notes: list[Realised]
    dropped: list[Item]
    slices: list[Slice]
    thumb_slices: set = field(default_factory=set)
    strum_ticks: set = field(default_factory=set)
    lh_positions: dict = field(default_factory=dict)  # slice tick -> index position

    @staticmethod
    def from_path(best: State, slices: list[Slice], items: list[Item], solver: Solver) -> "Solution":
        path = []
        st = best
        while st.parent is not None:
            path.append(st)
            st = st.parent
        path.reverse()
        by_id = {it.id: it for it in items}
        realised: dict[int, Realised] = {}
        dropped: list[Item] = []
        strums = set()
        thumbs = set()
        for si, (sl, stt) in enumerate(zip(slices, path)):
            c = stt.cand
            for it in sl.attacks:
                o = c.assign.get(it.id)
                if o is None or o.dropped:
                    dropped.append(it)
                    continue
                r = Realised(it, o, it.start, it.end, si)
                if c.flags.get("lh_tap") == it.id:
                    r.lh_tap = True
                if c.flags.get("strum"):
                    r.strum = True
                realised[it.id] = r
            for hid in c.dropped_holds:
                if hid in realised:
                    realised[hid].end = min(realised[hid].end, sl.tick)
            if c.flags.get("strum"):
                strums.add(sl.tick)
            if c.thumb:
                thumbs.add(si)
        sol = Solution(best.cost, sorted(realised.values(), key=lambda r: (r.start, -r.item.priority)), dropped, slices, thumbs, strums)
        assign_fingers(sol, path, solver)
        return sol


def assign_fingers(sol: Solution, path: list[State], solver: Solver) -> None:
    """Annotate left-hand fingers (1-4, T) and right-hand fingers (p i m a)."""
    by_slice: dict[int, list[Realised]] = {}
    for r in sol.notes:
        by_slice.setdefault(r.slice_index, []).append(r)
    sounding: list[Realised] = []
    prev_pos = None
    last_single = "m"
    held_finger: dict[int, str] = {}
    for si, sl in enumerate(sol.slices):
        attacks = by_slice.get(si, [])
        sounding = [r for r in sounding if r.end > sl.tick] + attacks
        pts = []
        for r in sounding:
            pl = r.option.place
            if pl is None or pl.kind in ("tap", "dead", "open"):
                continue
            lf = pl.lh_fret
            if lf is None or lf <= 0:
                continue
            if pl.kind == "nh" and r not in attacks:
                continue  # harmonics ring on after the touch lifts
            pts.append((lf, pl.string, r))
        if pts:
            lo = min(p[0] for p in pts)
            hi = max(p[0] for p in pts)
            span = comfortable_span(lo)
            lo_pos = max(1, hi - (span if hi - lo <= span else span + 1))
            pos = lo if prev_pos is None else min(max(prev_pos, lo_pos), lo)
            prev_pos = pos
            sol.lh_positions[sl.tick] = pos
            groups: dict[float, list] = {}
            for f, s, r in pts:
                groups.setdefault(f, []).append((s, r))
            used = set()
            barre_fret = None
            if len(groups.get(lo, [])) > 1:
                strs = [s for s, _ in groups[lo]]
                blocked = any(
                    (o.option.place and o.option.place.string in range(min(strs), max(strs) + 1) and o.option.place.kind == "open")
                    for o in sounding
                )
                if not blocked:
                    barre_fret = lo
            fingers_needed = sum(1 if (f == barre_fret) else len(v) for f, v in groups.items())
            for f in sorted(groups):
                members = sorted(groups[f], key=lambda x: -x[0])  # low string first
                if f == barre_fret:
                    for s, r in members:
                        r.lh_finger = r.lh_finger or "1"
                    used.add(1)
                    continue
                for s, r in members:
                    if r.item.id in held_finger:
                        r.lh_finger = held_finger[r.item.id]
                        used.add(int(r.lh_finger) if r.lh_finger.isdigit() else 0)
                        continue
                    want = int(min(4, max(1, f - pos + 1)))
                    while want in used and want < 4:
                        want += 1
                    while want in used and want > 1:
                        want -= 1
                    if want in used and s == 6:
                        r.lh_finger = "T"
                        continue
                    if s == 6 and si in sol.thumb_slices and fingers_needed > 4 and f <= lo + 1:
                        r.lh_finger = "T"
                        fingers_needed -= 1
                        continue
                    used.add(want)
                    r.lh_finger = str(want)
            for f, s, r in pts:
                held_finger[r.item.id] = r.lh_finger
        # right hand
        plucked = [r for r in attacks if r.option.place and r.option.place.kind in ("fret", "open", "nh", "ah")]
        taps = [r for r in attacks if r.option.place and r.option.place.kind == "tap"]
        for r in taps:
            r.rh_finger = "i"
        if taps:
            for r in plucked:
                r.lh_tap = True
            continue
        if not plucked:
            continue
        plucked.sort(key=lambda r: -r.option.place.string)
        ah = [r for r in plucked if r.option.place.kind == "ah"]
        if ah:
            ah[0].rh_finger = "a"
            for r in plucked:
                if r is not ah[0]:
                    r.rh_finger = "p"
            continue
        if any(r.strum for r in plucked):
            continue
        if len(plucked) == 1:
            r = plucked[0]
            if r.option.place.string >= 4 and r.item.role != "melody":
                r.rh_finger = "p"
            else:
                last_single = "i" if last_single == "m" else "m"
                r.rh_finger = last_single
            continue
        rest = plucked
        if plucked[0].option.place.string >= 3 or len(plucked) >= 4:
            plucked[0].rh_finger = "p"
            rest = plucked[1:]
            if len(rest) >= 4 and rest[0].option.place.string >= 4:
                rest[0].rh_finger = "p"
                rest = rest[1:]
        # Fingers keep their natural strings (i→3, m→2, a→1) whenever possible.
        order = ["i", "m", "a", "c"]
        start = max(0, min(len(order) - len(rest), 3 - rest[0].option.place.string)) if rest else 0
        for k, r in enumerate(rest):
            r.rh_finger = order[min(3, start + k)]
