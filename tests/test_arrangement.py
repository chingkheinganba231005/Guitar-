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


def test_a_low_sung_ballad_stays_in_its_octave_and_low_on_the_neck():
    """A Thousand Years-like case: a 6/8 ballad sung around Bb3-Bb4. The tune
    is played where it is sung (not an octave up at the 15th-20th fret), no
    fretted note goes above the 14th fret, and nothing Guitar I plays cuts a
    sustaining melody note short on its string."""
    song = demo_song("sixeight")
    for n in song.melody:
        n.pitch -= 7
    arr = arrange(song, ArrangeOptions())
    g1 = arr.guitars[0]
    assert g1.stats["melody_octave_shift"] == 0
    assert audit_project(to_atelier(arr)) == []
    for g in arr.guitars:
        assert all(n.fret + g.capo <= 14 for n in g.placed if isinstance(n.fret, (int, float)) and not n.harmonic)
    tune = [n for n in g1.placed if n.role == "melody"]
    assert len(tune) == len(song.melody)
    for m in tune:
        assert not any(n is not m and n.string == m.string and m.start < n.start < m.end for n in g1.placed)
    assert not any(n.harmonic == "natural" and n.fret == 5 for g in arr.guitars for n in g.placed)


def test_fills_guitar_one_cannot_hold_go_to_guitar_two():
    from duet_arranger.arrange.accomp import comp_part
    from duet_arranger.model import Note

    song = demo_song("ballad", bars_per_section=2)
    t = song.bars[3].start + 12
    fill = Note(t, t + 12, 69, role="handoff", priority=6.0)
    out = comp_part(song, [fill], (64, 59, 55, 50, 45, 40), 0)
    assert any(n.role == "fill" and n.start == t and n.pitch == 69 for n in out)


def _held_song():
    """The 6/8 ballad with one bar where the singer holds a note and the band
    holds the chord under it (like "one step clo-ser~")."""
    song = demo_song("sixeight")
    bar = song.bars[12]
    first = min((n for n in song.melody if bar.start <= n.start < bar.end), key=lambda n: n.start)
    song.melody = [n for n in song.melody if not (bar.start <= n.start < bar.end) or n is first]
    first.start, first.end = bar.start, bar.end
    for name, ns in song.parts.items():
        kept = []
        for n in ns:
            if bar.start < n.start < bar.end:
                continue
            if n.start == bar.start:
                n.end = max(n.end, bar.end)
            kept.append(n)
        song.parts[name] = kept
    song.bass = [n for n in song.bass if not (bar.start < n.start < bar.end)]
    return song, bar


def test_both_guitars_ring_a_held_chord():
    from duet_arranger.arrange.lead import held_spans

    song, bar = _held_song()
    assert any(a == bar.start and b >= bar.end for a, b in held_spans(song))
    arr = arrange(song, ArrangeOptions())
    assert audit_project(to_atelier(arr)) == []
    for g in arr.guitars:
        inside = [n for n in g.placed if bar.start < n.start < bar.end and n.fret != "x"]
        assert inside == [], (g.name, [(n.start - bar.start, n.role) for n in inside])
        assert any(n.start == bar.start for n in g.placed), g.name


def _strings_song(strings_level: float):
    """The 6/8 ballad with a string section: pads swelling in under the
    intro's piano chords (off the piano's attacks, as separated strings are
    transcribed), and a violin tune over the outro's chords. ``strings_level``
    is the strings stem's loudness relative to the piano."""
    from duet_arranger.model import Note

    song = demo_song("sixeight")
    strings = []
    for bar in song.bars[:8]:
        ch = song.chord_at(bar.start)
        for k, p in enumerate((58 + (ch.root - 10) % 12, 65)):
            st = bar.start + 7 + 5 * k
            strings.append(Note(st, bar.end + 6, p, velocity=0.5, source="other"))
    tune = [70, 72, 74, 72, 70, 69, 67, 69, 70, 74, 77, 75, 74, 72, 70, 70]
    k = 0
    for bar in song.bars[16:20]:
        for j, d in enumerate((18, 6, 12, 36)):
            st = bar.start + sum((18, 6, 12, 36)[:j])
            strings.append(Note(st, st + d, tune[k % len(tune)], velocity=0.7, source="other"))
            k += 1
    song.parts["other"] = strings
    song.notes_info = {"stem_rms": {"piano": 1.0, "other": strings_level}}
    return song


def test_string_pad_does_not_hide_the_intro_chords():
    from duet_arranger.arrange.lead import chordal_regions

    song = _strings_song(0.9)
    regions = chordal_regions(song, song.melody)
    assert any(a == 0 and b >= 8 for a, b in regions)


def test_guitar_one_plays_a_string_tune_between_sections():
    """Where only the strings sing (between a chorus and the next verse, or
    here over the outro), Guitar I plays their tune as its melody."""
    song = _strings_song(0.4)  # softer than the piano, as strings often are
    arr = arrange(song, ArrangeOptions())
    assert audit_project(to_atelier(arr)) == []
    lead = arr.guitars[0].placed
    outro = (song.bars[16].start, song.bars[19].end)
    tune = sorted({n.start for n in song.parts["other"] if outro[0] <= n.start < outro[1]})
    mel = {n.start for n in lead if n.role == "melody" and outro[0] <= n.start < outro[1]}
    assert len(mel & set(tune)) >= 0.9 * len(tune)
    assert not any(n.role == "chord" and outro[0] <= n.start < outro[1] for n in lead)
