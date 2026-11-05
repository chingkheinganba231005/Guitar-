import json

import pytest

from duet_arranger.arrange.duet import ArrangeOptions, arrange
from duet_arranger.demo import demo_song
from duet_arranger.export.ascii_tab import project_text
from duet_arranger.export.atelier import split_straight, to_atelier
from duet_arranger.arrange.lines import LineIndex
from duet_arranger.model import TPQ
from duet_arranger.validate import audit_project, bar_fill_errors

PERC = {"SB", "WT", "BW"}


@pytest.fixture(scope="module", params=["pop", "ballad"])
def arranged(request):
    song = demo_song(request.param)
    arr = arrange(song, ArrangeOptions())
    return request.param, song, arr, to_atelier(arr)


def test_two_tracks_and_valid_json(arranged):
    style, song, arr, p = arranged
    assert p["version"] == 2 and len(p["tracks"]) == 2
    json.dumps(p)
    for t in p["tracks"]:
        assert len(t["bars"]) == len(song.bars)
        assert len(t["tuning"]) == 6


def test_no_physically_impossible_passages(arranged):
    _, _, _, p = arranged
    assert audit_project(p) == []
    assert bar_fill_errors(p) == []


def test_melody_is_kept(arranged):
    _, song, arr, _ = arranged
    lead = arr.guitars[0]
    played = {(n.start, n.pitch % 12) for n in lead.placed if n.pitch is not None}
    mel = [n for n in song.melody]
    hit = sum(1 for n in mel if (n.start, n.pitch % 12) in played)
    assert hit / len(mel) >= 0.97


def test_percussion_only_for_drum_driven_songs(arranged):
    style, _, arr, _ = arranged
    techs = {t for g in arr.guitars for n in g.placed for t in n.techs}
    if style == "pop":
        assert techs & PERC
    else:
        assert not (techs & PERC)


def test_capo_only_when_necessary(arranged):
    _, _, arr, _ = arranged
    assert all(g.capo == 0 for g in arr.guitars)


def test_ties_reference_previous_event(arranged):
    _, _, _, p = arranged
    for t in p["tracks"]:
        prev = None
        for b in t["bars"]:
            for e in b["events"]:
                for n in e["notes"]:
                    if n.get("tie"):
                        assert n["tieFrom"] == prev["id"]
                        assert any(m["string"] == n["string"] and m["fret"] != "x" for m in prev["notes"])
                prev = e


def sung_bars(song, arr):
    """Bars where Guitar I carries the tune (not the ones where it strikes the band's chords)."""
    lead = arr.guitars[0].placed
    chord_bars = {song.bar_at(n.start) for n in lead if n.role == "chord"}
    return sorted({song.bar_at(n.start) for n in lead if n.role == "melody"} - chord_bars), chord_bars


def test_guitar_one_is_a_complete_fingerstyle_part(arranged):
    """Guitar I plays like a solo fingerstyle guitar: the tune, its own bass
    under it and harmony in between."""
    _, song, arr, _ = arranged
    lead = arr.guitars[0].placed
    bars, _ = sung_bars(song, arr)
    assert bars
    with_bass = [bi for bi in bars if any(n.role == "bass" and n.start == song.bars[bi].start for n in lead)]
    assert len(with_bass) >= 0.9 * len(bars)
    mel = LineIndex([n for n in lead if n.role == "melody"])
    for n in lead:
        m = mel.at(n.start)
        if n.role == "bass" and m is not None:
            assert m.pitch - n.pitch >= 5, (n.start, n.pitch, m.pitch)
    harmony = [n for n in lead if n.role in ("support", "inner") and song.bar_at(n.start) in set(bars)]
    assert len(harmony) >= len(bars)


def test_guitar_two_never_doubles_guitar_one(arranged):
    _, song, arr, _ = arranged
    g1, g2 = arr.guitars
    _, chord_bars = sung_bars(song, arr)
    attacks = {(n.start, n.pitch) for n in g1.placed if n.pitch is not None and n.fret != "x"}
    doubled = [
        n
        for n in g2.placed
        if n.pitch is not None and n.fret != "x" and (n.start, n.pitch) in attacks and not (n.role == "bass" and song.bar_at(n.start) in chord_bars)
    ]
    assert doubled == []
    # and Guitar II still has plenty to play where Guitar I sings
    bars, _ = sung_bars(song, arr)
    busy = {song.bar_at(n.start) for n in g2.placed if n.fret != "x"}
    assert len(busy & set(bars)) >= 0.9 * len(bars)


def test_guitar_two_doubles_in_octaves_not_unisons(arranged):
    """Doubling is welcome in another octave: the tune in octaves in the
    choruses, the bass an octave lower on chord changes. Never in unison."""
    from duet_arranger.arrange.accomp import chorus_bars

    _, song, arr, _ = arranged
    g1, g2 = arr.guitars
    choruses = chorus_bars(song)
    assert choruses
    g2_attacks = {(n.start, n.pitch) for n in g2.placed if n.pitch is not None and n.fret != "x"}
    tune = [n for n in g1.placed if n.role == "melody" and song.bar_at(n.start) in choruses]
    octaved = [m for m in tune if (m.start, m.pitch - 12) in g2_attacks or (m.start, m.pitch + 12) in g2_attacks]
    assert len(octaved) >= 0.7 * len(tune)
    outside = [n for n in g2.placed if n.role == "octave" and song.bar_at(n.start) not in choruses]
    assert outside == []


def test_octaves_can_be_switched_off():
    song = demo_song("ballad", bars_per_section=2)
    arr = arrange(song, ArrangeOptions(octaves=False))
    assert not any(n.role == "octave" for n in arr.guitars[1].placed)


def test_six_eight_guitar_one_keeps_the_eighths_moving():
    """In 6/8 Guitar I has its bass on beats 1 and 4 and rolls chord tones on
    the eighths while the tune holds, like a solo 6/8 fingerstyle piece."""
    song = demo_song("sixeight")
    arr = arrange(song, ArrangeOptions())
    assert audit_project(to_atelier(arr)) == []
    lead = arr.guitars[0].placed
    bars, _ = sung_bars(song, arr)
    beat4 = [bi for bi in bars if any(n.role == "bass" and n.start == song.bars[bi].start + song.bars[bi].ticks_per_beat for n in lead)]
    assert len(beat4) >= 0.8 * len(bars)
    between = [n for n in lead if n.role == "inner" and (n.start - song.bars[song.bar_at(n.start)].start) % song.bars[0].ticks_per_beat]
    assert len(between) >= len(bars)


def test_ballad_uses_harmonics(arranged):
    style, _, arr, _ = arranged
    if style == "ballad":
        assert sum(1 for g in arr.guitars for n in g.placed if n.harmonic) >= 3


def test_ascii_tab_renders(arranged):
    _, _, _, p = arranged
    txt = project_text(p)
    assert "Guitar I" in txt and "Guitar II" in txt


def test_split_straight():
    assert split_straight(0, 24, TPQ) == [(24, 4, 0)]
    assert split_straight(12, 24, TPQ) == [(12, 8, 0), (12, 8, 0)]
    assert split_straight(0, 36, TPQ) == [(36, 4, 1)]
    assert sum(L for L, _, _ in split_straight(6, 90, TPQ)) == 90


def test_forced_tuning_and_capo():
    song = demo_song("pop", bars_per_section=2)
    arr = arrange(song, ArrangeOptions(tunings="DADGAD,Open G", capo="2"))
    assert arr.guitars[0].tuning.name == "DADGAD" and arr.guitars[1].tuning.name == "Open G"
    assert all(g.capo == 2 for g in arr.guitars)
    assert audit_project(to_atelier(arr)) == []
