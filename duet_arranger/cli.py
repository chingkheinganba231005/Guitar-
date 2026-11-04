"""duet-arranger: song in, two-guitar fingerstyle arrangement out."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

from .guitar import NAMED_TUNINGS, STANDARD, Tuning

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif", ".opus", ".wma"}
MIDI_EXT = {".mid", ".midi"}
NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)?$")


def parse_tuning(spec: str) -> Tuning:
    """A tuning name, or six notes low→high ("D A D G A D", "D2 A2 D3 G3 A3 D4", "38 45 50 55 57 62")."""
    for k, v in NAMED_TUNINGS.items():
        if k.lower() == spec.strip().lower():
            return Tuning(k, v)
    parts = re.split(r"[\s,\-]+", spec.strip())
    if len(parts) == 1 and len(parts[0]) == 6 and parts[0].isalpha():
        parts = list(parts[0])
    if len(parts) != 6:
        raise ValueError(f"Tuning needs six notes low→high or a name: {spec!r}")
    std_low_high = list(reversed(STANDARD))
    pitches = []
    for i, p in enumerate(parts):
        if p.lstrip("-").isdigit():
            pitches.append(int(p))
            continue
        m = NOTE_RE.match(p)
        if not m:
            raise ValueError(f"Not a note: {p!r}")
        pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
        if m.group(3) is not None:
            pitches.append(pc + 12 * (int(m.group(3)) + 1))
        else:
            ref = std_low_high[i]
            best = min((ref + d for d in range(-8, 6) if (ref + d) % 12 == pc % 12), key=lambda x: abs(x - ref))
            pitches.append(best)
    t = Tuning(spec, tuple(reversed(pitches)))
    if t.tension_penalty() is None:
        raise ValueError(f"Tuning {spec!r} puts unsafe tension on a string (max +2 semitones, +1 on the high E)")
    return t


def load_song(path: str, args) -> "Song":
    ext = os.path.splitext(path)[1].lower()
    if ext in MIDI_EXT:
        from .midi_input import read_midi

        song = read_midi(path, args.title)
    elif ext in AUDIO_EXT or ext == "":
        from .audio.pipeline import analyze_audio

        song = analyze_audio(path, args.output, title=args.title, artist=args.artist or "", device=args.device)
    else:
        raise SystemExit(f"Unsupported input type {ext!r}. Use audio (wav/mp3/flac/…) or MIDI.")
    if args.artist:
        song.artist = args.artist
    return song


def write_outputs(arr, out_dir: str, base: str, title: str | None, artist: str | None, progress=None) -> dict:
    from .export.ascii_tab import project_text
    from .export.atelier import to_atelier
    from .export.midi_out import write_midi
    from .validate import audit_project, bar_fill_errors

    from .progress import report as _report

    os.makedirs(out_dir, exist_ok=True)
    _report(progress, "export", 0.1, "Writing the Atelier duet")
    project = to_atelier(arr, title=title, artist=artist)
    _report(progress, "export", 0.5, "Checking every hand position")
    issues = audit_project(project)
    fill = bar_fill_errors(project)
    project["arranger"]["playability_issues"] = [str(i) for i in issues]
    paths = {
        "atelier": os.path.join(out_dir, f"{base}.atelier.json"),
        "tab": os.path.join(out_dir, f"{base}.tab.txt"),
        "midi": os.path.join(out_dir, f"{base}.preview.mid"),
        "report": os.path.join(out_dir, f"{base}.report.json"),
    }
    with open(paths["atelier"], "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False, indent=1)
    with open(paths["tab"], "w", encoding="utf-8") as f:
        f.write(project_text(project))
    write_midi(arr, paths["midi"])
    report = dict(arr.report)
    report["playability_issues"] = [str(i) for i in issues]
    report["incomplete_bars"] = fill
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    _report(progress, "export", 1.0, "Done")
    return {"paths": paths, "issues": issues, "fill": fill, "project": project}


def summarise(arr, result, seconds: float) -> str:
    lines = []
    song = arr.song
    from .theory import pc_name, prefers_flats

    key = f"{pc_name(song.key_tonic, prefers_flats(song.key_tonic, song.key_mode))} {song.key_mode}"
    lines.append(f"Song: {song.title} · {key} · {len(song.bars)} bars · {round(song.bars[0].tempo)} BPM · {'drum-driven (percussive parts on)' if song.percussive else 'melodic (no percussion)'}")
    for g in arr.guitars:
        st = g.stats
        lines.append(
            f"  {g.name}: {g.tuning.name} [{g.tuning.label}]{' · capo ' + str(g.capo) if g.capo else ' · no capo'}"
            f" · {st.get('notes', 0)} notes, {st.get('harmonics', 0)} harmonics, {st.get('open_strings', 0)} open strings, {st.get('dropped', 0)} dropped"
        )
    n = len(result["issues"])
    lines.append(f"Playability audit: {'no issues' if n == 0 else str(n) + ' issue(s) – see report'}")
    for k, p in result["paths"].items():
        lines.append(f"  {k:8s} {p}")
    lines.append(f"Done in {seconds:.0f}s. Open the .atelier.json in Atelier with File → Import Atelier project.")
    return "\n".join(lines)


def cmd_arrange(args) -> int:
    from .arrange.duet import ArrangeOptions, arrange

    t0 = time.time()
    song = load_song(args.input, args)
    opts = ArrangeOptions(
        tunings=args.tuning,
        capo=args.capo,
        max_capo=args.max_capo,
        max_fret=args.max_fret,
        percussion=args.percussion,
        harmonics=not args.no_harmonics,
        taps=not args.no_taps,
        difficulty=args.difficulty,
    )
    arr = arrange(song, opts)
    base = re.sub(r"[^A-Za-z0-9 _-]", "", args.title or song.title).strip().replace(" ", "-") or "arrangement"
    result = write_outputs(arr, args.output, base, args.title, args.artist)
    print(summarise(arr, result, time.time() - t0))
    return 0


def cmd_demo(args) -> int:
    from .arrange.duet import ArrangeOptions, arrange
    from .demo import demo_song

    t0 = time.time()
    song = demo_song(args.style)
    arr = arrange(song, ArrangeOptions(tunings=args.tuning, capo=args.capo))
    result = write_outputs(arr, args.output, f"demo-{args.style}", None, None)
    print(summarise(arr, result, time.time() - t0))
    return 0


def cmd_tunings(args) -> int:
    for name, p in NAMED_TUNINGS.items():
        t = Tuning(name, p)
        print(f"{name:24s} {t.label}")
    return 0


def cmd_audit(args) -> int:
    from .validate import audit_project, bar_fill_errors

    with open(args.project, encoding="utf-8") as f:
        project = json.load(f)
    issues = audit_project(project, args.max_fret)
    for i in issues:
        print(i)
    for e in bar_fill_errors(project):
        print("incomplete:", e)
    print(f"{len(issues)} playability issue(s)")
    return 1 if issues else 0


def cmd_serve(args) -> int:
    try:
        from .web.server import serve
    except ImportError as err:  # pragma: no cover - depends on the install
        raise SystemExit(f"The web app needs FastAPI and uvicorn: pip install -e \".[web]\" ({err})")
    serve(args.host, args.port, args.jobs, args.atelier, args.token, args.open, args.max_upload_mb)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="duet-arranger", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arrange", help="arrange an audio or MIDI file for two guitars")
    a.add_argument("input")
    a.add_argument("-o", "--output", default="output")
    a.add_argument("--title")
    a.add_argument("--artist")
    a.add_argument("--tuning", default="auto", help='auto, standard, a name, six notes ("D A D G A D"), or two separated by a comma for Guitar I,Guitar II')
    a.add_argument("--capo", default="auto", help="auto (only when it clearly helps), none, or a fret number")
    a.add_argument("--max-capo", type=int, default=7)
    a.add_argument("--max-fret", type=int, default=20, help="highest fret your guitar reaches comfortably (cutaway ≈ 20)")
    a.add_argument("--percussion", default="auto", choices=["auto", "on", "off"])
    a.add_argument("--no-harmonics", action="store_true")
    a.add_argument("--no-taps", action="store_true")
    a.add_argument("--difficulty", type=float, default=1.0, help="0.6 lighter … 1.0 full texture")
    a.add_argument("--device", default=None, help="torch device for separation (cpu, cuda, mps)")
    a.set_defaults(func=cmd_arrange)
    d = sub.add_parser("demo", help="arrange the built-in demo song")
    d.add_argument("style", nargs="?", default="pop", choices=["pop", "ballad"])
    d.add_argument("-o", "--output", default="output")
    d.add_argument("--tuning", default="auto")
    d.add_argument("--capo", default="auto")
    d.set_defaults(func=cmd_demo)
    w = sub.add_parser("serve", help="run the web app (upload a song in your browser)")
    w.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other devices")
    w.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    w.add_argument("--jobs", default=os.environ.get("DUET_JOBS_DIR", "duet-jobs"), help="folder for uploads and results")
    w.add_argument("--atelier", default=None, help="folder of an Atelier checkout to serve, or the URL of a hosted Atelier")
    w.add_argument("--token", default=None, help="require this access token (for a server others can reach)")
    w.add_argument("--max-upload-mb", type=int, default=300)
    w.add_argument("--open", action="store_true", help="open the browser")
    w.set_defaults(func=cmd_serve)
    t = sub.add_parser("tunings", help="list named tunings")
    t.set_defaults(func=cmd_tunings)
    v = sub.add_parser("audit", help="check an Atelier project for physically impossible passages")
    v.add_argument("project")
    v.add_argument("--max-fret", type=int, default=20)
    v.set_defaults(func=cmd_audit)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
