from duet_arranger.fingering import Solver, SolverConfig
from duet_arranger.guitar import STANDARD, sounding_pitch
from duet_arranger.model import TPQ, Note


def sec(t):
    return t / TPQ * 0.5


def test_scale_melody_with_bass_is_fully_played_and_correct():
    notes = []
    for i, p in enumerate([67, 69, 71, 72, 74, 76, 78, 79] * 2):
        notes.append(Note(i * 12, (i + 1) * 12, p, role="melody", priority=100))
    for b in range(2):
        notes.append(Note(b * 96, (b + 1) * 96, 43, role="bass", priority=20))
    sol = Solver(STANDARD, 0, SolverConfig(), sec).solve(notes)
    assert not [d for d in sol.dropped if d.role == "melody"]
    for r in sol.notes:
        assert sounding_pitch(r.option.place, STANDARD, 0) == r.option.pitch == r.item.pitch + r.option.shift


def test_out_of_reach_note_uses_tap_or_harmonic_or_octave():
    # Bass held at fret 3 on string 6 while the melody leaps to fret 15 on string 1
    notes = [Note(0, 96, 43, role="bass", priority=20), Note(12, 24, 79, role="melody", priority=100), Note(24, 48, 64 + 15, role="melody", priority=100)]
    sol = Solver(STANDARD, 0, SolverConfig(), sec).solve(notes)
    assert not [d for d in sol.dropped if d.role == "melody"]


def test_capo_frets_are_relative():
    notes = [Note(0, 24, 66, role="melody", priority=100)]
    sol = Solver(STANDARD, 2, SolverConfig(), sec).solve(notes)
    r = sol.notes[0]
    assert sounding_pitch(r.option.place, STANDARD, 2) == 66


def test_no_fifth_fret_harmonic_with_fretted_notes():
    from duet_arranger.fingering import lh_eval

    cfg = SolverConfig()
    # a 5th-fret harmonic with open strings is fine, with a fretted note it is not
    assert lh_eval([(3, 5, "touch")], {3: -1, 2: 0}, cfg) is not None
    assert lh_eval([(3, 5, "touch"), (2, 7, "press")], {3: -1, 2: 7}, cfg) is None
    # a 12th-fret harmonic next to a fretted note stays possible
    assert lh_eval([(3, 12, "touch"), (2, 12, "press")], {3: -1, 2: 12}, cfg) is not None


def test_tune_avoids_high_frets():
    """A tune that fits low on the neck is not played up at the 15th fret."""
    notes = [Note(i * TPQ, (i + 1) * TPQ, p, role="melody", priority=100.0) for i, p in enumerate([76, 74, 72, 71, 69, 67])]
    sol = Solver(STANDARD, 0, SolverConfig(), lambda t: t / TPQ * 0.6).solve(notes)
    frets = [r.option.place.fret for r in sol.notes if r.option.place.kind == "fret"]
    assert frets and max(frets) <= 12
