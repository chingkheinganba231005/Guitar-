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
    if style == "waltz":
        return waltz_song(bars_per_section)
    if style in ("sixeight", "6/8"):
        return sixeight_song(bars_per_section)
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


# An original slow waltz in Bb: a piano intro of chords on every beat, then a
# sung verse over eighth-note piano arpeggios (the texture of many ballads).
WALTZ_CHORDS = [(10, "maj"), (5, "maj"), (7, "min"), (3, "maj")]  # Bb F Gm Eb
WALTZ_TUNE = [  # (offset in eighths, length in eighths, pitch) per bar of 3/4
    [(0, 2, 65), (2, 2, 70), (4, 2, 72)],
    [(0, 4, 74), (4, 2, 72)],
    [(0, 2, 70), (2, 1, 69), (3, 1, 70), (4, 2, 74)],
    [(0, 6, 72)],
]


def waltz_song(bars_per_section: int = 4) -> Song:
    tempo = 136.0
    song = Song(title="Demo waltz", artist="Duet Arranger", key_tonic=10, key_mode="major")
    bar_len = 3 * TPQ
    t = 0
    melody, bass, piano = [], [], []
    energy = []
    layout = [("Intro", False), ("Intro", False), ("Verse", True), ("Verse", True), ("Outro", False)]
    for label, sung in layout:
        if not song.sections or song.sections[-1].label != label:
            song.sections.append(Section(len(song.bars), label))
        for k in range(bars_per_section):
            song.bars.append(Bar(t, 3, 4, tempo, TPQ))
            root, q = WALTZ_CHORDS[k % 4]
            song.chords.append(Chord(t, t + bar_len, root, q))
            pcs = chord_pcs(root, q)
            energy.append(0.35 if not sung else 0.6)
            broot = 34 + ((root - 10) % 12)
            if broot > 45:
                broot -= 12
            bass.append(Note(t, t + bar_len, broot, velocity=0.6, source="bass"))
            voicing = sorted(58 + ((pc - 10) % 12) for pc in pcs)
            if not sung:
                for beat in range(3):  # a chord on every beat
                    s0 = t + beat * TPQ
                    for p in voicing:
                        piano.append(Note(s0, s0 + TPQ, p, velocity=0.6, source="piano"))
            else:
                arp = [voicing[0], voicing[1], voicing[2], voicing[1] + 12, voicing[2], voicing[1]]
                for e8, p in enumerate(arp):
                    s0 = t + e8 * TPQ // 2
                    piano.append(Note(s0, s0 + TPQ // 2, p, velocity=0.5, source="piano"))
                for off, length, p in WALTZ_TUNE[k % 4]:
                    s0 = t + off * TPQ // 2
                    melody.append(Note(s0, s0 + length * TPQ // 2, p, velocity=0.75, source="vocals"))
            t += bar_len
    song.melody = melody
    song.bass = bass
    song.parts = {"vocals": list(melody), "bass": list(bass), "piano": piano}
    song.energy = energy
    song.percussive = False
    return song


# An original 6/8 ballad in Bb (dotted quarter = 50): piano chords on each
# dotted quarter in the intro, then eighth-note left-hand arpeggios under a
# sung melody — the texture of many slow triple-time pop ballads.
SIX8_INTRO = [(10, "maj"), (10, "maj"), (7, "min"), (7, "min"), (3, "maj"), (3, "maj"), (10, "maj"), (5, "maj")]
SIX8_VERSE = [(3, "maj"), (10, "maj"), (7, "min"), (5, "maj")]
SIX8_TUNE = [  # (offset in eighths, length in eighths, pitch), six eighths per bar
    [(0, 3, 70), (3, 2, 70), (5, 1, 72)],
    [(0, 3, 74), (3, 3, 72)],
    [(0, 2, 70), (2, 1, 69), (3, 3, 67)],
    [(0, 6, 65)],
]


def sixeight_song(bars_per_section: int = 4) -> Song:
    tempo = 75.0  # quarter-note BPM: dotted quarter = 50
    song = Song(title="Demo six-eight", artist="Duet Arranger", key_tonic=10, key_mode="major")
    L = 72
    t = 0
    melody, bass, piano, energy = [], [], [], []
    plan = [("Intro", c, False) for c in SIX8_INTRO]
    plan += [("Verse", SIX8_VERSE[k % 4], True) for k in range(2 * bars_per_section)]
    plan += [("Outro", c, False) for c in SIX8_INTRO[:4]]
    verse_k = 0
    for label, (root, q), sung in plan:
        if not song.sections or song.sections[-1].label != label:
            song.sections.append(Section(len(song.bars), label))
        song.bars.append(Bar(t, 6, 8, tempo, 36))
        song.chords.append(Chord(t, t + L, root, q))
        pcs = chord_pcs(root, q)
        energy.append(0.6 if sung else 0.35)
        broot = 34 + ((root - 10) % 12)
        if broot > 41:
            broot -= 12
        voicing = sorted(62 + ((pc - 2) % 12) for pc in pcs)
        if not sung:
            for h in (0, 36):  # a chord and a bass note on each dotted quarter
                for p in voicing:
                    piano.append(Note(t + h, t + h + 36, p, velocity=0.6, source="piano"))
                bass.append(Note(t + h, t + h + 36, broot + 12, velocity=0.6, source="bass"))
        else:
            third = [p for p in range(broot + 12, broot + 30) if p % 12 == pcs[1]][0]
            arp = [broot, broot + 7, broot + 12, third, broot + 12, broot + 7]
            for e8, p in enumerate(arp):
                piano.append(Note(t + 12 * e8, t + 12 * e8 + 12, p + 12, velocity=0.5, source="piano"))
            bass.append(Note(t, t + L, broot, velocity=0.6, source="bass"))
            for off, length, p in SIX8_TUNE[verse_k % 4]:
                melody.append(Note(t + 12 * off, t + 12 * (off + length), p, velocity=0.75, source="vocals"))
            verse_k += 1
        t += L
    song.melody = melody
    song.bass = bass
    song.parts = {"vocals": list(melody), "bass": list(bass), "piano": piano}
    song.energy = energy
    song.percussive = False
    return song
