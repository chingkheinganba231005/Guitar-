"""Top level: decide tunings and capos, build both guitar parts, finger them."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..fingering import Solution, Solver, SolverConfig, note_options, Item
from ..guitar import NATURAL_HARMONICS, STANDARD, Tuning, key_shifted_tunings
from ..model import TPQ, Hit, Note, Song
from ..progress import report
from ..quantize import snap_notes
from ..theory import chord_pcs, scale_pcs
from .accomp import comp_part
from .lead import lead_part, melody_shift_candidates
from .percussion import percussion_hits
from .lines import LineIndex
from .resonance import resonance_additions
from .techniques import PlacedNote, add_expression, placed_notes

log = logging.getLogger(__name__)


@dataclass
class ArrangeOptions:
    tunings: str = "auto"  # auto | standard | <tuning name> | "name1,name2" per guitar
    capo: str = "auto"  # auto | none | <int>
    max_capo: int = 7
    capo_bias: float = 0.4  # a capo must save roughly this share of the effort to be used
    max_fret: int = 20
    percussion: str = "auto"  # auto | on | off
    harmonics: bool = True
    taps: bool = True
    difficulty: float = 1.0  # 0.6 = lighter texture, 1.0 = full
    screen_top: int = 3
    beam: int = 28
    verbose: bool = False


@dataclass
class GuitarArrangement:
    name: str
    role: str
    tuning: Tuning
    capo: int
    content: list[Note]
    hits: list[Hit]
    solution: Solution
    placed: list[PlacedNote] = field(default_factory=list)
    score: float = 0.0
    stats: dict = field(default_factory=dict)


@dataclass
class Arrangement:
    song: Song
    guitars: list[GuitarArrangement]
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------- content


def build_content(song: Song, role: str, tuning, capo: int, lead_notes: list[Note] | None, opts: ArrangeOptions, shift: int | None = None):
    if role == "lead":
        notes = lead_part(song, tuning, capo, opts.difficulty, shift)
        hits: list[Hit] = []
    else:
        notes = comp_part(song, lead_notes or [], tuning, capo)
        hits = percussion_hits(song, opts.difficulty)
    notes = clip_repeats(snap_notes(notes, song))
    return notes, hits


def clip_repeats(notes: list[Note]) -> list[Note]:
    """A pitch struck again ends its previous instance (that is how strings behave)."""
    by_pitch: dict[int, list[Note]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    out = []
    for group in by_pitch.values():
        group.sort(key=lambda n: (n.start, -n.priority))
        kept: list[Note] = []
        for n in group:
            if kept and kept[-1].start == n.start:
                continue
            if kept and kept[-1].end > n.start:
                kept[-1].end = n.start
            kept.append(n)
        out += kept
    return sorted(out, key=lambda n: (n.start, -n.priority))


def _fresh(notes: list[Note]) -> list[Note]:
    """Copies with new ids so repeated solves never share mutable state."""
    out = []
    for n in notes:
        c = n.copy()
        out.append(c)
    return out


# ---------------------------------------------------------------- scoring


def static_score(content: list[Note], tuning, capo: int, cfg: SolverConfig, song: Song, role: str) -> tuple[float, float]:
    """Cheap per-note playability and resonance estimate used to pre-screen tunings."""
    total = 0.0
    for n in content:
        it = Item(note=n)
        opts = note_options(it, tuning, capo, cfg)
        total += min(o.cost for o in opts)
    total /= max(1, len(content))
    return total, resonance_bonus(tuning, capo, song, role)


def capo_penalty(per_note: float, capo: int, opts: ArrangeOptions) -> float:
    """A capo has to earn its place: it must save a clear share of the effort."""
    if not capo:
        return 0.0
    return opts.capo_bias * max(per_note, 0.4) + 0.06 + 0.012 * capo


def register_bonus(content: list[Note], role: str) -> float:
    """Reward a lead melody that sits clearly above the accompaniment."""
    if role != "lead":
        return 0.0
    mel = sorted(n.pitch for n in content if n.role == "melody")
    if not mel:
        return 0.0
    med = mel[len(mel) // 2]
    return 0.3 * min(1.0, max(0.0, (med - 57) / 10))


def resonance_bonus(tuning, capo: int, song: Song, role: str) -> float:
    """Open strings that belong to the key (and its main chords) ring sympathetically."""
    scale = set(scale_pcs(song.key_tonic, song.key_mode))
    tonic_chord = set(chord_pcs(song.key_tonic, "maj" if song.key_mode == "major" else "min"))
    common = {}
    for c in song.chords:
        common[(c.root, c.quality)] = common.get((c.root, c.quality), 0) + (c.end - c.start)
    top = sorted(common.items(), key=lambda kv: -kv[1])[:4]
    top_pcs = set()
    for (r, q), _ in top:
        top_pcs |= set(chord_pcs(r, q))
    bonus = 0.0
    for p in tuning:
        q = (p + capo) % 12
        if q in tonic_chord:
            bonus += 0.05
        elif q in top_pcs:
            bonus += 0.035
        elif q in scale:
            bonus += 0.015
        else:
            bonus -= 0.05
    if role == "accomp":
        lowest = (min(tuning) + capo) % 12
        if lowest == song.key_tonic:
            bonus += 0.06
    return bonus * (1.3 if not song.percussive else 1.0)


def tuning_penalty(t: Tuning, capo: int, opts: ArrangeOptions) -> float:
    pen = t.tension_penalty() or 0.0
    return 0.012 * pen + 0.018 * t.changed_strings()


def solution_score(sol: Solution, content: list[Note], t: Tuning, capo: int, song: Song, role: str, opts: ArrangeOptions, melody=None) -> tuple[float, dict]:
    n = max(1, len(content))
    per_note = sol.cost / n
    opens = sum(1 for r in sol.notes if r.option.place and r.option.place.kind == "open")
    harmonics = sum(1 for r in sol.notes if r.option.place and r.option.place.kind in ("nh", "ah"))
    dropped = [it for it in sol.dropped if it.kind == "note"]
    mel_dropped = sum(1 for it in dropped if it.role == "melody")
    bonus = resonance_bonus(t.pitches, capo, song, role) + register_bonus(content, role)
    added = 0
    if role == "accomp":
        bonus += 0.25 * opens / n
        added = len(resonance_additions(placed_notes(sol, song, t.pitches, capo), song, t.pitches, capo, melody))
        bonus += (0.45 if song.percussive else 0.7) * added / n
    if not song.percussive:
        bonus += 0.4 * harmonics / n
    score = per_note - bonus + tuning_penalty(t, capo, opts) + capo_penalty(per_note, capo, opts)
    stats = {
        "cost_per_note": round(per_note, 3),
        "notes": len(content),
        "open_strings": opens,
        "harmonics": harmonics,
        "dropped": len(dropped),
        "melody_dropped": mel_dropped,
        "resonance": round(bonus, 3),
        "open_string_resonance_notes": added,
    }
    return score, stats


# ---------------------------------------------------------------- search


def candidate_tunings(song: Song, opts: ArrangeOptions, index: int) -> list[Tuning]:
    from ..guitar import NAMED_TUNINGS

    spec = opts.tunings.strip()
    if spec == "auto":
        return key_shifted_tunings(song.key_tonic, song.key_mode)
    if spec == "standard":
        return [Tuning("Standard", STANDARD)]
    names = [s.strip() for s in spec.split(",")]
    name = names[min(index, len(names) - 1)]
    if name.lower() == "auto":
        return key_shifted_tunings(song.key_tonic, song.key_mode)
    for k, v in NAMED_TUNINGS.items():
        if k.lower() == name.lower():
            return [Tuning(k, v)]
    # custom pitches "D A D G A D" low->high or MIDI numbers
    from ..cli import parse_tuning

    return [parse_tuning(name)]


def candidate_capos(opts: ArrangeOptions) -> list[int]:
    if opts.capo == "none":
        return [0]
    if opts.capo not in ("auto", ""):
        return [int(opts.capo)]
    return list(range(0, opts.max_capo + 1))


def sample_content(content: list[Note], hits: list[Hit], song: Song, max_bars: int = 20):
    if len(song.bars) <= max_bars:
        return content, hits
    step = len(song.bars) / max_bars
    keep = set()
    for k in range(max_bars):
        b = int(k * step)
        keep.add(b)
        keep.add(min(len(song.bars) - 1, b + 1))
    c = [n for n in content if song.bar_at(n.start) in keep]
    h = [x for x in hits if song.bar_at(x.start) in keep]
    return c, h


def choose_and_solve(song: Song, role: str, name: str, opts: ArrangeOptions, lead_notes: list[Note] | None, index: int, progress=None) -> GuitarArrangement:
    stage = "guitar1" if role == "lead" else "guitar2"
    report(progress, stage, 0.0, f"{name}: trying every tuning and capo")
    cfg_full = SolverConfig(max_fret=opts.max_fret, beam=opts.beam, per_key=18, allow_taps=opts.taps, allow_harmonics=opts.harmonics, percussive=bool(song.percussive and role == "accomp"))
    cfg_fast = SolverConfig(max_fret=opts.max_fret, beam=6, per_key=6, allow_taps=opts.taps, allow_harmonics=opts.harmonics, percussive=cfg_full.percussive)
    tunings = candidate_tunings(song, opts, index)
    capos = candidate_capos(opts)
    shifts = melody_shift_candidates(song, 2) if role == "lead" else [None]
    mel_index = LineIndex([n for n in (lead_notes or []) if n.role == "melody"])
    t0 = time.time()
    # Stage 1: static screen of every tuning x capo (x melody octave)
    screened = []
    for t in tunings:
        for capo in capos:
            for shift in shifts:
                content, hits = build_content(song, role, t.pitches, capo, lead_notes, opts, shift)
                per_note, bonus = static_score(content, t.pitches, capo, cfg_fast, song, role)
                bonus += register_bonus(content, role)
                s = per_note - bonus + tuning_penalty(t, capo, opts) + capo_penalty(per_note, capo, opts)
                screened.append((s, t, capo, shift))
    screened.sort(key=lambda x: x[0])
    report(progress, stage, 0.15, f"{name}: {len(screened)} set-ups screened, fingering the best")
    stage2 = screened[: max(6, opts.screen_top * 3)]
    for t in tunings:  # standard tuning without capo always gets a real hearing
        if t.pitches == STANDARD and 0 in capos:
            for shift in shifts:
                if not any(x[1].pitches == STANDARD and x[2] == 0 and x[3] == shift for x in stage2):
                    stage2.append((0.0, t, 0, shift))
    # Stage 2: fast search on a sample of bars
    fast = []
    for k2, (s, t, capo, shift) in enumerate(stage2):
        report(progress, stage, 0.15 + 0.3 * k2 / max(1, len(stage2)), f"{name}: quick fingering in {t.name}{' capo ' + str(capo) if capo else ''}")
        content, hits = build_content(song, role, t.pitches, capo, lead_notes, opts, shift)
        c, h = sample_content(content, hits, song)
        try:
            sol = Solver(t.pitches, capo, cfg_fast, song.seconds_at).solve(_fresh(c), h)
        except RuntimeError:
            continue
        sc, _ = solution_score(sol, c, t, capo, song, role, opts, mel_index)
        fast.append((sc, t, capo, shift))
    fast.sort(key=lambda x: x[0])
    finalists = fast[: opts.screen_top] or [(0, Tuning("Standard", STANDARD), 0, shifts[0])]
    # Always let standard tuning without capo compete: it is the default to beat.
    std = [x for x in fast if x[1].pitches == STANDARD and x[2] == 0]
    if std and not any(t.pitches == STANDARD and c == 0 for _, t, c, _ in finalists):
        finalists.append(std[0])
    best = None
    considered = []
    for k3, (_, t, capo, shift) in enumerate(finalists):
        report(progress, stage, 0.45 + 0.55 * k3 / max(1, len(finalists)), f"{name}: full fingering in {t.name}{' capo ' + str(capo) if capo else ''}")
        content, hits = build_content(song, role, t.pitches, capo, lead_notes, opts, shift)
        try:
            sol = Solver(t.pitches, capo, cfg_full, song.seconds_at).solve(content, hits)
        except RuntimeError as err:
            log.warning("%s: %s capo %s unplayable: %s", name, t.name, capo, err)
            continue
        sc, stats = solution_score(sol, content, t, capo, song, role, opts, mel_index)
        if shift is not None:
            stats["melody_octave_shift"] = shift
        considered.append({"tuning": t.name, "pitches": t.label, "capo": capo, "score": round(sc, 3), **stats})
        if best is None or sc < best.score:
            best = GuitarArrangement(name, role, t, capo, content, hits, sol, score=sc, stats=stats)
    if best is None:
        raise RuntimeError(f"Could not find a playable arrangement for {name}")
    best.stats["considered"] = considered
    best.stats["search_seconds"] = round(time.time() - t0, 1)
    best.stats["screened"] = len(screened)
    log.info("%s: %s capo %d (score %.3f) in %.1fs", name, best.tuning.name, best.capo, best.score, time.time() - t0)
    report(progress, stage, 1.0, f"{name}: {best.tuning.name}{' · capo ' + str(best.capo) if best.capo else ' · no capo'}")
    return best


def arrange(song: Song, opts: ArrangeOptions | None = None, progress=None) -> Arrangement:
    opts = opts or ArrangeOptions()
    if opts.percussion == "on":
        song.percussive = True
    elif opts.percussion == "off":
        song.percussive = False
    lead = choose_and_solve(song, "lead", "Guitar I", opts, None, 0, progress)
    lead_notes = [n for n in lead.content]
    accomp = choose_and_solve(song, "accomp", "Guitar II", opts, lead_notes, 1, progress)
    mel_index = LineIndex([n for n in lead_notes if n.role == "melody"])
    for g in (lead, accomp):
        g.placed = placed_notes(g.solution, song, g.tuning.pitches, g.capo)
        add_expression(g.placed, song)
    accomp.placed += resonance_additions(accomp.placed, song, accomp.tuning.pitches, accomp.capo, mel_index)
    accomp.placed.sort(key=lambda n: (n.start, n.string))
    report = {
        "key": f"{song.key_tonic} {song.key_mode}",
        "percussive": song.percussive,
        "drum_prominence": round(song.drum_prominence, 3),
        "guitars": [
            {
                "name": g.name,
                "role": g.role,
                "tuning": g.tuning.name,
                "tuning_pitches": g.tuning.label,
                "capo": g.capo,
                "score": round(g.score, 3),
                **{k: v for k, v in g.stats.items()},
            }
            for g in (lead, accomp)
        ],
    }
    return Arrangement(song, [lead, accomp], report)
