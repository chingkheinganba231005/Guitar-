"""Regressions from the first real-song test (a 3/4 ballad with a chordal piano intro)."""

import random
from collections import Counter

import pytest

from duet_arranger.arrange.duet import ArrangeOptions, arrange
from duet_arranger.audio.rhythm import smooth_tempos
from duet_arranger.demo import demo_song
from duet_arranger.export.atelier import to_atelier
from duet_arranger.model import TPQ, Bar, Note
from duet_arranger.quantize import snap_notes
from duet_arranger.validate import audit_project, bar_fill_errors


def noisy(song, seed=3):
    """What a transcription does to a clean performance: wobbly onsets, stray notes."""
    rng = random.Random(seed)
    for name, notes in song.parts.items():
        out = []
        for n in notes:
            j = rng.uniform(-0.11, 0.11) * TPQ
            m = n.copy(start=max(0, n.start + j), end=n.end + rng.uniform(-0.2, 0.1) * TPQ)
            out.append(m)
            if name == "piano" and rng.random() < 0.08:  # a spurious extra onset
                out.append(n.copy(start=n.start + rng.choice([3, 9, 15]), end=n.start + 18))
        song.parts[name] = snap_notes(out, song)
    song.melody = list(song.parts["vocals"])
    song.bass = list(song.parts["bass"])
    return song


@pytest.fixture(scope="module")
def waltz():
    song = noisy(demo_song("waltz"))
    arr = arrange(song, ArrangeOptions())
    return song, arr, to_atelier(arr)


def events(track):
    return [e for b in track["bars"] for e in b["events"]]


def test_no_32nd_notes_and_playable(waltz):
    _, _, p = waltz
    assert audit_project(p) == [] and bar_fill_errors(p) == []
    for t in p["tracks"]:
        dens = Counter(e["den"] for e in events(t))
        assert dens.get(32, 0) == 0 and dens.get(64, 0) == 0, (t["name"], dens)


def test_let_ring_and_few_rests(waltz):
    _, _, p = waltz
    assert p["defaultLetRing"] is True
    for t in p["tracks"]:
        attacks = [n for e in events(t) for n in e["notes"] if not n.get("tie")]
        rung = sum(1 for n in attacks if n["sustain"]) / len(attacks)
        assert rung > 0.9, (t["name"], rung)
        rests = sum(1 for e in events(t) if not e["notes"])
        assert rests <= 2, (t["name"], rests)


def test_intro_is_chords_on_the_beat_in_guitar_one(waltz):
    song, arr, p = waltz
    intro_end = song.bars[8].start
    lead = [n for n in arr.guitars[0].placed if n.start < intro_end]
    assert lead and all(n.role in ("chord", "bass", "resonance") for n in lead)
    onsets = sorted({n.start for n in lead})
    assert all(t % TPQ == 0 for t in onsets)  # on the beats, like the piano
    assert len(onsets) >= 20
    by_tick = Counter(n.start for n in lead)
    assert sum(1 for c in by_tick.values() if c >= 3) >= 0.8 * len(by_tick)  # full chords


def test_accompaniment_has_one_regular_pattern_per_section(waltz):
    song, arr, _ = waltz
    acc = arr.guitars[1].placed
    verse = range(8, 16)
    shapes = set()
    for bi in verse:
        b = song.bars[bi]
        shapes.add(tuple(sorted({n.start - b.start for n in acc if b.start <= n.start < b.end})))
    assert len(shapes) <= 2, shapes
    assert all(all(off % (TPQ // 2) == 0 for off in sh) for sh in shapes)


def test_harmonics_prefer_12th_and_7th(waltz):
    _, arr, _ = waltz
    frets = Counter(n.fret for g in arr.guitars for n in g.placed if n.harmonic == "natural")
    assert frets.get(5, 0) + frets.get(4, 0) + frets.get(9, 0) <= max(1, 0.2 * sum(frets.values()))


def test_tempo_wobble_is_not_written():
    bars = [Bar(i * 72, 3, 4, [144, 136][i % 2], TPQ) for i in range(40)]
    smooth_tempos(bars)
    assert len({b.tempo for b in bars}) == 1


def test_snap_prefers_eighths():
    song = demo_song("waltz")
    ns = snap_notes([Note(13, 20, 60), Note(4, 10, 60), Note(7, 15, 60)], song)
    assert [n.start for n in ns] == [12, 6, 6]
