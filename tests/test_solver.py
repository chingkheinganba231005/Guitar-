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
