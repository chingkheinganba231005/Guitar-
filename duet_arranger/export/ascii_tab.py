"""Plain-text tablature for quick reading in a terminal or text editor.

Legend: 7 fretted · <12> natural harmonic · [5] artificial harmonic ·
t9 right-hand tap · x muted/percussive · h/p hammer/pull · s legato slide ·
/ \\ slides · ~ vibrato · ( ) held from before · * thumb slap/body hit.
"""

from __future__ import annotations

from ..theory import note_name

WIDTH = {1: 16, 2: 8, 4: 4, 8: 3, 16: 2, 32: 2, 64: 2}


def _cell(n: dict) -> str:
    f = n["fret"]
    if n.get("tie"):
        return ""
    tech = n.get("tech", [])
    if f == "x":
        s = "x"
    else:
        fs = str(int(f)) if float(f).is_integer() else str(f)
        if n.get("harmonic") == "natural":
            s = f"<{fs}>"
        elif n.get("harmonic"):
            s = f"[{fs}]"
        elif "RH" in tech:
            s = f"t{fs}"
        else:
            s = fs
    for t, mark in (("H", "h"), ("P", "p"), ("LS", "s"), ("/", "/"), ("\\", "\\"), ("V", "~")):
        if t in tech:
            s += mark
    if any(t in tech for t in ("SB", "WT", "BW")):
        s += "*"
    return s


def track_text(track: dict, project: dict, bars_per_line: int = 4) -> str:
    tuning = track["tuning"]
    names = [note_name(p)[:-1].ljust(2) for p in tuning]
    lines_out = []
    header = f"{track['name']} — {track.get('role', '')}".strip(" —")
    tun = " ".join(note_name(p)[:-1] for p in reversed(tuning))
    lines_out.append(f"{header}\nTuning (low→high): {tun}{'   Capo ' + str(track['capo']) if track.get('capo') else ''}")
    bars = track["bars"]
    for start in range(0, len(bars), bars_per_line):
        rows = [names[s] + "|" for s in range(6)]
        top = "   "
        chords = "   "
        for bi in range(start, min(start + bars_per_line, len(bars))):
            b = bars[bi]
            if b.get("section"):
                top_mark = f"[{b['section']}]"
            else:
                top_mark = str(bi + 1)
            seg_top = top_mark
            seg_ch = ""
            cols = []
            for e in b["events"]:
                cells = {n["string"]: _cell(n) for n in e["notes"]}
                w = max([WIDTH.get(e["den"], 2)] + [len(c) + 1 for c in cells.values()])
                if e.get("chord"):
                    w = max(w, len(e["chord"]) + 1)
                cols.append((w, cells, e))
            bar_w = sum(w for w, _, _ in cols) + 1
            ch_line = ""
            for w, cells, e in cols:
                ch_line += (e.get("chord") or "").ljust(w)
                for s in range(6):
                    c = cells.get(s + 1, "")
                    rows[s] += (c if c else "-").ljust(w, "-")
            for s in range(6):
                rows[s] += "-|"
            top += seg_top[: bar_w].ljust(bar_w + 1)
            chords += ch_line.ljust(bar_w + 1)
        lines_out.append("")
        lines_out.append(top.rstrip())
        if chords.strip():
            lines_out.append(chords.rstrip())
        lines_out.extend(rows)
    return "\n".join(lines_out)


def project_text(project: dict, bars_per_line: int = 4) -> str:
    out = [project.get("title", ""), project.get("subtitle", ""), project.get("composer", ""), ""]
    out.append(__doc__.split("\n\n", 1)[1].strip())
    for t in project["tracks"]:
        out.append("")
        out.append("=" * 72)
        out.append(track_text(t, project, bars_per_line))
    return "\n".join(out) + "\n"
