"""stem-studio: separate a song or video and follow every instrument."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def cmd_serve(args) -> int:
    try:
        from .server import serve
    except ImportError as err:  # pragma: no cover - depends on the install
        raise SystemExit(f'Stem Studio needs FastAPI and uvicorn: pip install -e ".[web]" ({err})')
    serve(args.host, args.port, args.jobs, args.token, args.open, args.max_upload_mb, args.device)
    return 0


def cmd_analyse(args) -> int:
    from .analyze import analyse

    res = analyse(args.input, args.output, progress=lambda s, f, m: print(f"[{s} {f * 100:3.0f}%] {m}", flush=True), device=args.device)
    print(f"{res['key']} · {res['meter']} · {res['tempo_label']} · {len(res['bars'])} bars")
    for t in res["tracks"]:
        what = "silent" if t["silent"] else f"{len(t['notes'])} notes, {len(t['chords'])} chords" + (f", {len(t['hits'])} hits" if t["hits"] else "")
        print(f"  {t['label']:16s} {what}")
    print(f"Written to {os.path.abspath(args.output)} (analysis.json, audio/)")
    return 0


def main(argv=None) -> int:
    _force_utf8()
    p = argparse.ArgumentParser(prog="stem-studio", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("serve", help="run the web app")
    w.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from other devices")
    w.add_argument("--port", type=int, default=int(os.environ.get("STUDIO_PORT", 8001)))
    w.add_argument("--jobs", default=os.environ.get("STUDIO_JOBS_DIR", "studio-jobs"), help="folder for songs and their analysis")
    w.add_argument("--token", default=None, help="require this access token")
    w.add_argument("--max-upload-mb", type=int, default=1024)
    w.add_argument("--device", default=None, help="torch device for separation (cpu, cuda, mps)")
    w.add_argument("--open", action="store_true", help="open the browser")
    w.set_defaults(func=cmd_serve)
    a = sub.add_parser("analyse", help="analyse one file without the web app")
    a.add_argument("input")
    a.add_argument("-o", "--output", default="studio-output")
    a.add_argument("--device", default=None)
    a.set_defaults(func=cmd_analyse)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
