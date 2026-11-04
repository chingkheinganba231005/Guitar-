import json

import pytest

from duet_arranger.arrange.duet import ArrangeOptions, arrange
from duet_arranger.demo import demo_song
from duet_arranger.export.ascii_tab import project_text
from duet_arranger.export.atelier import split_straight, to_atelier
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
