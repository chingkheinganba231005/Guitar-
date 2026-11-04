# Duet Arranger

Give it a song. It separates the recording into vocals, drums, bass, guitar,
piano and other instruments, transcribes each part, and writes a
**fingerstyle arrangement for two guitars** that stays as close to the
original as two pairs of hands allow. The tabs open directly in
[Atelier](https://github.com/chingkheinganba231005/atelier-fingerstyle), where
both guitars appear together in one score.

```
song.mp3 ──► Demucs stems ──► Basic Pitch notes ──► beat / key / chords / sections
         ──► Guitar I (melody) + Guitar II (bass, harmony, groove)
         ──► tuning & capo search ──► physical fingering search ──► Atelier duet
```

## What it decides for you

**Who plays what.** Guitar I sings: the vocal melody (or the instrumental hook
where nobody sings), with chord tones a third or sixth beneath important
notes, and the original instrumental fills in the melody's rests. Guitar II is
the band: the real bass line, and an accompaniment that copies the original
comping rhythm — chords where the piano or guitar strikes chords, the same
arpeggio where they arpeggiate. Only where the source holds long pads (which a
guitar cannot sustain) does it fall back on an idiomatic fingerstyle pattern.

**Percussion only when the song needs it.** Drum prominence is measured from
the separated drum stem. In drum-driven songs, kick hits become heel-of-hand
body thumps (BW), snare hits become thumb slaps on the bass note (SB), wrist
thumps on a chord (WT) or palm slaps on muted strings. Ballads and soft songs
get no percussion at all.

**Tunings that ring.** Every tuning is considered — standard, drop D, double
drop D, DADGAD, open D, open D minor, open G, open G minor, DGDGAD, DGDGCD,
CGDGBD, open C, open C6, open E, open A, drop C, D and E♭ standard — plus
open tunings re-pitched to the song's key (never more than two semitones above
standard on any string, one on the high E). Each guitar gets its own tuning.
Tunings win when they make the part easier *and* when their open strings
belong to the song: free open strings that are chord tones are added to chord
attacks so the guitar rings on every string it can.

**Harmonics where the song is soft.** Long, exposed melody notes and quiet
intros/outros try natural harmonics on real nodes of the chosen tuning
(12th, 7th/19th, 5th, 4th/9th frets), or artificial (harp) harmonics. Guitar
II's arpeggios turn into harmonic arpeggios in soft passages when the tuning
has the notes on a node.

**A capo only when it is really necessary.** Every capo position is tried,
but a capo is only chosen when it removes a large share of the effort (about
40%); otherwise the arrangement stays open.

**Nothing a hand cannot do.** Every onset is solved with a beam search over
realisations, enforcing:

* left hand: four fingers (plus thumb over the neck on the 6th string), fret
  spans that grow higher up the neck, barres that never cover a string that
  must ring open, held notes keeping their finger down, finger-crossing costs;
* right hand: four plucking fingers (five with a thumb brush, more only as a
  strum or rolled chord), one right-hand tap at a time with no other plucks
  (the left hand may hammer a note from nowhere meanwhile), artificial
  harmonics with at most one thumb note beneath;
* movement: big shifts in little time are expensive, and impossible jumps are
  refused.

Notes that are out of the left hand's reach can become right-hand taps. An
independent audit then re-reads the written Atelier file and reports any
event that would still be unplayable (the report lists them; the test suite
requires zero).

**Expression from the recording.** Fast legato lines get hammer-ons and
pull-offs, connected vocal leaps get legato slides, long sung notes get
vibrato, vocal scoops and falls become slide-in (SI) and slide-out (SO),
syncopated hits in drum-driven songs get accents. Every note carries a
left-hand finger (1–4, T) and a right-hand finger (p i m a).

## Install

Python 3.10 or newer and [ffmpeg](https://ffmpeg.org/) (for MP3/M4A input).

```bash
git clone https://github.com/chingkheinganba231005/Guitar-.git duet-arranger
cd duet-arranger
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .                                        # core (MIDI input, arranging, export)
pip install -r requirements-audio.txt                   # audio input: Demucs, ONNX runtime …
pip install --no-deps basic-pitch                       # Basic Pitch without its TensorFlow pin
```

The first audio run downloads the Demucs `htdemucs_6s` model (~80 MB).
Separation runs on the GPU when PyTorch sees one (`--device cuda` / `mps`),
otherwise on the CPU (roughly half real time on four cores).

## Use

```bash
duet-arranger arrange "My Song.mp3" -o output --title "My Song" --artist "Someone"
```

Output (in `output/`):

| File | What it is |
| --- | --- |
| `My-Song.atelier.json` | The duet for Atelier: **File → Import Atelier project** |
| `My-Song.tab.txt` | Plain-text tab of both guitars |
| `My-Song.preview.mid` | MIDI preview of the two guitars |
| `My-Song.report.json` | Tunings, capos, alternatives considered, playability audit |
| `stems/*.wav` | The separated vocals, drums, bass, guitar, piano and other |
| `analysis.json` | Detected tempo, meter, key, chords, sections, drum prominence |

A multi-track MIDI file works as input too (`duet-arranger arrange song.mid`);
tracks named like *vocal/melody*, *bass*, *piano*, *guitar* and channel-10
drums are recognised automatically.

### Options

| Option | Default | |
| --- | --- | --- |
| `--tuning` | `auto` | `standard`, a name (`DADGAD`, `"Open G"`), six notes low→high (`"C G D G B D"`), or one per guitar separated by a comma: `"Standard,Open D"` |
| `--capo` | `auto` | `none`, or a fret number to force it |
| `--percussion` | `auto` | `on` / `off` overrides the drum-prominence decision |
| `--max-fret` | `20` | highest fret your guitar reaches comfortably |
| `--no-harmonics`, `--no-taps` | | leave those techniques out |
| `--difficulty` | `1.0` | `0.6` gives lighter textures |
| `--device` | auto | `cpu`, `cuda`, `mps` for separation |

Other commands:

```bash
duet-arranger demo pop -o output      # arrange the built-in demo song (pop or ballad)
duet-arranger tunings                 # list named tunings
duet-arranger audit file.atelier.json # check any Atelier project for unplayable passages
```

## Atelier

Atelier gained what the arranger writes: multiple guitars on one shared bar
structure (full-score and single-guitar views, per-guitar tuning, capo, volume,
pan, mute and solo, multi-part MusicXML, PDF of the view), chord symbols,
left/right-hand fingering, accents, slide-in/slide-out and the extra open
tunings. Open the `.atelier.json` with **File → Import Atelier project**.

## How it works (code map)

| Module | |
| --- | --- |
| `audio/separate.py` | Demucs separation into six stems |
| `audio/transcribe.py` | Basic Pitch per stem; vocal scoops/falls from pitch bends |
| `audio/rhythm.py` | beat tracking, tempo-octave correction, meter and downbeats |
| `audio/drums.py` | kick/snare/hat detection, drum prominence |
| `analysis.py` | key, chords (template + Viterbi), sections, energy |
| `midi_input.py` | MIDI files in |
| `arrange/lead.py` | Guitar I: melody, support notes, fills, harmonics |
| `arrange/accomp.py` | Guitar II: bass, imitated comping, patterns |
| `arrange/percussion.py` | percussive layer for drum-driven songs |
| `arrange/resonance.py` | free open strings at chord attacks |
| `arrange/duet.py` | tuning × capo × melody-octave search |
| `fingering.py` | the physical model and beam search |
| `arrange/techniques.py` | legato, slides, vibrato, scoops, accents, taps |
| `export/atelier.py` | Atelier v2 writer (durations, tuplets, ties) |
| `validate.py` | independent playability audit |

## Tests

```bash
pip install -e ".[dev]"
pytest                          # fretboard rules, solver, arrangements, MIDI round trip
DUET_AUDIO_TESTS=1 pytest       # also runs Demucs + Basic Pitch on a synthesised song
```

## Limits worth knowing

* Transcription quality bounds everything. Dense mixes, heavy reverb or
  doubled vocals give messier melody lines; the arranger fingers whatever it
  is given, so check the melody in Atelier and fix stray notes there.
* Downbeat detection is heuristic (chord changes, kick and bass accents). If a
  song is shifted by a beat, the arrangement is still correct musically but
  barlines are misplaced.
* Each Atelier guitar has one rhythmic voice; sustained bass under a moving
  melody is written with ties.
* Ballads with rubato get per-bar tempo changes rather than a free tempo map.
