"""A small original song used for demos and tests (no copyrighted material).

``demo_song("pop")`` has a drum kit, syncopated piano chords and a busy bass;
``demo_song("ballad")`` has no drums, sustained pads and a slow, long-note
melody. Both share the same tune and chord progression.
"""

from __future__ import annotations

from .model import TPQ, Bar, Chord, Hit, Note, Section, Song
from .theory import chord_pcs

PROGRESSION = [(7, "maj"), (2, "maj"), (4, "min"), (0, "maj")]  # G D Em C

# melody per bar: (offset in 8ths, length in 8ths, pitch)
TUNE = [
    [(0, 2, 71), (2, 1, 74), (3, 1, 76), (4, 3, 74), (7, 1, 71)],
    [(0, 3, 69), (3, 1, 66), (4, 2, 69), (6, 2, 74)],
    [(0, 2, 71), (2, 2, 67), (4, 1, 71), (5, 1, 74), (6, 2, 79)],
    [(0, 6, 76), (6, 1, 74), (7, 1, 72)],
]
CHORUS = [
    [(0, 1, 79), (1, 1, 78), (2, 2, 79), (4, 2, 81), (6, 2, 83)],
    [(0, 3, 81), (3, 1, 78), (4, 4, 74)],
    [(0, 1, 76), (1, 1, 78), (2, 2, 79), (4, 2, 83), (6, 2, 86)],
    [(0, 8, 84)],
]


def demo_song(style: str = "pop", bars_per_section: int = 4) -> Song:
    pop = style == "pop"
    tempo = 96.0 if pop else 68.0
    song = Song(title=f"Demo {style}", artist="Duet Arranger", key_tonic=7, key_mode="major")
    layout = [("Intro", None), ("Verse", TUNE), ("Chorus", CHORUS), ("Verse", TUNE), ("Chorus", CHORUS), ("Outro", None)]
    t = 0
    bar_len = 4 * TPQ
    melody, bass, piano, pads = [], [], [], []
    hits = []
    energy = []
    for si, (label, tune) in enumerate(layout):
        song.sections.append(Section(len(song.bars), label))
        for k in range(bars_per_section):
            song.bars.append(Bar(t, 4, 4, tempo, TPQ))
            root, q = PROGRESSION[k % 4]
            song.chords.append(Chord(t, t + bar_len, root, q))
            pcs = chord_pcs(root, q)
            e = {"Intro": 0.3, "Verse": 0.5, "Chorus": 0.85, "Outro": 0.25}[label]
            energy.append(e)
            if tune is not None:
                for off, length, p in tune[k % 4]:
                    s = t + off * TPQ // 2
                    melody.append(Note(s, s + length * TPQ // 2, p - 12 if not pop else p - 12, velocity=0.75 + 0.15 * (label == "Chorus"), source="vocals"))
            elif label == "Intro":
                # instrumental hook in the intro (piano right hand)
                for off, length, p in TUNE[k % 4][:3]:
                    s = t + off * TPQ // 2
                    piano.append(Note(s, s + length * TPQ // 2, p + 12, velocity=0.7, source="piano"))
            broot = 43 + ((root - 7) % 12)
            if broot > 50:
                broot -= 12
            if pop:
                for eighth in (0, 3, 4, 6):
                    s = t + eighth * TPQ // 2
                    p = broot if eighth != 6 else broot + 7
                    bass.append(Note(s, s + TPQ // 2, p, velocity=0.8, source="bass"))
                # syncopated piano chords: 1, 2&, 4
                for eighth, length in ((0, 3), (3, 3), (6, 2)):
                    s = t + eighth * TPQ // 2
                    for p in sorted(60 + ((pc - 0) % 12) for pc in pcs):
                        piano.append(Note(s, s + length * TPQ // 2, p, velocity=0.65, source="piano"))
                for beat in range(4):
                    s = t + beat * TPQ
                    hits.append(Hit(s, "kick" if beat in (0, 2) else "snare", 0.85))
                    hits.append(Hit(s + TPQ // 2, "hat", 0.4))
                hits.append(Hit(t + 7 * TPQ // 2, "kick", 0.6))
            else:
                bass.append(Note(t, t + bar_len, broot, velocity=0.6, source="bass"))
                for p in sorted(57 + ((pc - 9) % 12) for pc in pcs):
                    pads.append(Note(t, t + bar_len, p, velocity=0.5, source="other"))
            t += bar_len
    song.melody = melody
    song.bass = bass
    song.parts = {"vocals": list(melody), "bass": list(bass), "piano": piano}
    if pads:
        song.parts["other"] = pads
    song.hits = hits
    song.percussive = pop
    song.drum_prominence = 0.6 if pop else 0.0
    song.energy = energy
    return song
