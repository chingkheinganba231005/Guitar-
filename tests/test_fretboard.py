from duet_arranger.fingering import SolverConfig, lh_eval, rh_eval, Item, Option
from duet_arranger.guitar import NAMED_TUNINGS, Placement, Tuning, key_shifted_tunings, sounding_pitch
from duet_arranger.model import Note
from duet_arranger.cli import parse_tuning

CFG = SolverConfig()


def test_named_tunings_are_safe():
    for name, p in NAMED_TUNINGS.items():
        assert Tuning(name, p).tension_penalty() is not None, name


def test_key_shifted_tunings_never_overtighten():
    for tonic in range(12):
        for mode in ("major", "minor"):
            for t in key_shifted_tunings(tonic, mode):
                assert t.tension_penalty() is not None
                assert all(p - s <= 2 for p, s in zip(t.pitches, (64, 59, 55, 50, 45, 40)))


def test_harmonic_pitches():
    std = (64, 59, 55, 50, 45, 40)
    assert sounding_pitch(Placement(6, 12, "nh"), std, 0) == 52
    assert sounding_pitch(Placement(1, 7, "nh"), std, 0) == 83
    assert sounding_pitch(Placement(2, 3, "ah"), std, 2) == 59 + 2 + 3 + 12


def test_open_c_chord_shape_is_fine():
    # x32010
    pts = [(5, 3, "press"), (4, 2, "press"), (2, 1, "press")]
    played = {5: 3, 4: 2, 3: 0, 2: 1, 1: 0}
    assert lh_eval(pts, played, CFG) is not None


def test_barre_chord_ok_but_blocked_by_open_string():
    # F barre 133211
    pts = [(6, 1, "press"), (5, 3, "press"), (4, 3, "press"), (3, 2, "press"), (2, 1, "press"), (1, 1, "press")]
    played = {6: 1, 5: 3, 4: 3, 3: 2, 2: 1, 1: 1}
    assert lh_eval(pts, played, CFG) is not None
    # same but string 3 must ring open under the barre: impossible with 4 fingers
    pts2 = [(6, 1, "press"), (5, 3, "press"), (4, 3, "press"), (2, 1, "press"), (1, 1, "press")]
    played2 = {6: 1, 5: 3, 4: 3, 3: 0, 2: 1, 1: 1}
    res = lh_eval(pts2, played2, CFG)
    good = lh_eval(pts, played, CFG)
    assert res is None or res[0] > good[0] + 1.0  # contorted: three separate fingers at fret 1


def test_impossible_stretch():
    assert lh_eval([(6, 1, "press"), (1, 7, "press")], {6: 1, 1: 7}, CFG) is None
    assert lh_eval([(6, 12, "press"), (1, 17, "press")], {6: 12, 1: 17}, CFG) is not None


def test_right_hand_rules():
    def opt(s, f, kind):
        it = Item(note=Note(0, 24, 60, role="melody"))
        return it, Option((Placement(s, f, kind),), 0.0)

    assert rh_eval([opt(1, 12, "tap"), opt(5, 0, "open")], False) is None  # tap + pluck
    assert rh_eval([opt(1, 12, "tap"), opt(4, 5, "fret")], False) is not None  # tap + LH hammer
    assert rh_eval([opt(2, 5, "ah"), opt(6, 0, "open"), opt(4, 0, "open")], False) is None
    assert rh_eval([opt(2, 5, "ah"), opt(6, 0, "open")], False) is not None
    six = [opt(s, 0, "open") for s in range(1, 7)]
    cost, flags = rh_eval(six, False)
    assert flags.get("strum")


def test_parse_tuning():
    assert parse_tuning("DADGAD").pitches == (62, 57, 55, 50, 45, 38)
    assert parse_tuning("D A D G A D").pitches == (62, 57, 55, 50, 45, 38)
    assert parse_tuning("C G C G C E").pitches == (64, 60, 55, 48, 43, 36)
    assert parse_tuning("38 45 50 55 59 64").pitches == (64, 59, 55, 50, 45, 38)
