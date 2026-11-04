"""Percussive layer for drum-driven songs.

Kick drum hits become heel-of-hand body thumps (BW) carried by a plucked
note, or a muted low string; snare hits become a thumb slap on the bass
note (SB), a wrist thump on a chord (WT) or a palm slap on muted strings.
Ballads and other songs where the drums are not a main part get none.
"""

from __future__ import annotations

from ..model import Hit, Song


def percussion_hits(song: Song, density: float = 1.0) -> list[Hit]:
    if not song.percussive:
        return []
    out: list[Hit] = []
    seen = set()
    for h in sorted(song.hits, key=lambda h: h.start):
        if h.kind not in ("kick", "snare"):
            continue
        key = (h.start, h.kind)
        if key in seen:
            continue
        seen.add(key)
        if h.velocity < 0.25:
            continue  # ghost notes stay in the drummer's kit
        out.append(Hit(h.start, h.kind, h.velocity))
    if density < 1.0:
        out = [h for h in out if h.kind == "snare" or h.velocity > 0.5]
    return out
