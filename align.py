#!/usr/bin/env python3
"""
align.py — materialise etymon→reflex phonetic alignments and sound-change labels.

For every reflex (cldf/forms.csv) we align its form to its etymon (the head-word of its
parameter, cldf/parameters.csv) segment-by-segment and label each step with the sound change it
represents (spirantization, deaffrication, nasalization, cluster reduction, loss, …).

Segmentation reuses the curated CDIAL grapheme inventory (conversion/cdial-post.txt) — the same
profile make_cldf.py / models2 use — via longest-match tokenisation. Alignment is a
feature-weighted Needleman–Wunsch; labelling is a small articulatory-feature rule set.

Run after unify_cldf.py. Each final graph node is aligned to its final Origin_ID, so curated
borrowings, redirects, promoted section forms, and reparented descendants use the ancestry shown
by the application.

Output: cldf/alignments.csv, a NORMALISED (queryable) table — one row per aligned column:
    Form_ID, Origin_ID, Pos, Etymon_Idx, Etymon_Seg, Reflex_Seg, Change, Prev_Seg, Next_Seg

This is a computed, approximate layer (no hand-aligned gold standard yet); it's tuned to read
well on Indo-Aryan and degrade gracefully. A hand-override table can later be merged in here.

Usage:
    python align.py [--family INDO-ARYAN|ALL] [--limit N] [--out cldf/alignments.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from collections import defaultdict

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PROFILE = "conversion/cdial-post.txt"
FORMS = "cldf/forms.csv"

# Proto/reconstruction languages whose parameter Name is a true ancestor of the reflexes.
PROTO_LANGS = {"Indo-Aryan", "PDr", "PMu", "PNur", "PA", "PIA", "OIA"}


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation — longest-match over the curated grapheme inventory
# ─────────────────────────────────────────────────────────────────────────────

def load_graphemes(path: str) -> list[str]:
    gs: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and line.lower().startswith("grapheme"):
                continue
            g = line.rstrip("\n").split("\t")[0]
            if g:
                gs.add(unicodedata.normalize("NFC", g))
    # longest graphemes first so ʣʰ beats ʣ, ᵐb beats b, ā̃ beats ā
    return sorted(gs, key=len, reverse=True)


def segmenter(graphemes: list[str]):
    def tok(form: str) -> list[str]:
        # a hyphen marks a stem/affix boundary — strip it but keep the material on both sides
        s = unicodedata.normalize("NFC", form).replace("-", "").split("/")[0].strip()
        out: list[str] = []
        i = 0
        while i < len(s):
            for g in graphemes:
                if g and s.startswith(g, i):
                    out.append(g)
                    i += len(g)
                    break
            else:
                out.append(s[i])
                i += 1
        # drop pure punctuation / boundary marks
        return [g for g in out if g.strip() and g not in {",", ".", "*"}]
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# Articulatory features
# ─────────────────────────────────────────────────────────────────────────────

# base symbol → (place, manner, voice)
C = {
    "p": ("labial", "stop", 0), "b": ("labial", "stop", 1), "ɓ": ("labial", "stop", 1),
    "f": ("labial", "fricative", 0), "v": ("labial", "approximant", 1), "w": ("labial", "approximant", 1),
    "m": ("labial", "nasal", 1),
    "t": ("dental", "stop", 0), "d": ("dental", "stop", 1), "n": ("dental", "nasal", 1),
    "s": ("dental", "fricative", 0), "z": ("dental", "fricative", 1), "ẓ": ("dental", "fricative", 1),
    "r": ("dental", "trill", 1), "l": ("dental", "lateral", 1), "ɬ": ("dental", "lateral", 0),
    "ʦ": ("dental", "affricate", 0), "ʣ": ("dental", "affricate", 1),
    "c": ("palatal", "affricate", 0), "j": ("palatal", "affricate", 1),
    "ɟ": ("palatal", "stop", 1), "ʄ": ("palatal", "stop", 1),
    "ʧ": ("palatal", "affricate", 0), "ʤ": ("palatal", "affricate", 1),
    "ɲ": ("palatal", "nasal", 1), "ñ": ("palatal", "nasal", 1), "y": ("palatal", "approximant", 1),
    "ś": ("palatal", "fricative", 0), "ʃ": ("palatal", "fricative", 0), "ź": ("palatal", "fricative", 1),
    "k": ("velar", "stop", 0), "g": ("velar", "stop", 1), "ɠ": ("velar", "stop", 1),
    "ŋ": ("velar", "nasal", 1), "x": ("velar", "fricative", 0), "ɣ": ("velar", "fricative", 1),
    "h": ("glottal", "fricative", 0), "ɦ": ("glottal", "fricative", 1), "ḣ": ("glottal", "fricative", 0),
}
# retroflex counterparts (base letter + dot-below)
RETRO = {"t": "ʈ", "d": "ɖ", "n": "ɳ", "s": "ṣ", "r": "ɽ", "l": "ɭ", "z": "ẓ"}
RETRO_FEAT = {
    "ʈ": ("retroflex", "stop", 0), "ɖ": ("retroflex", "stop", 1), "ɳ": ("retroflex", "nasal", 1),
    "ṣ": ("retroflex", "fricative", 0), "ɽ": ("retroflex", "trill", 1), "ɭ": ("retroflex", "lateral", 1),
    "ṛ": ("retroflex", "trill", 1), "ḷ": ("retroflex", "lateral", 1), "ᶑ": ("retroflex", "stop", 1),
}
C.update(RETRO_FEAT)

V = {
    "a": ("low", "central", 0), "e": ("mid", "front", 0), "i": ("high", "front", 0),
    "o": ("mid", "back", 1), "u": ("high", "back", 1), "ə": ("mid", "central", 0),
    "ɛ": ("mid-low", "front", 0), "ε": ("mid-low", "front", 0), "ɔ": ("mid-low", "back", 1),
    "æ": ("low", "front", 0), "ɐ": ("low", "central", 0), "ʌ": ("mid", "central", 0),
    "ʊ": ("high", "back", 1), "ɪ": ("high", "front", 0),
}


class Seg:
    __slots__ = ("raw", "base", "kind", "place", "manner", "voice", "height", "back",
                 "long", "nasal", "aspirated", "prenasal", "retroflex", "idx")

    def __init__(self, raw: str):
        self.raw = raw
        self.idx = -1  # stable index within its own form (set for etymon segments)
        self.kind = "?"
        self.place = self.manner = self.height = self.back = None
        self.voice = self.long = self.nasal = self.aspirated = self.prenasal = self.retroflex = False

        s = raw
        if s and s[0] in ("ᵐ", "ⁿ"):
            self.prenasal = True
            s = s[1:]
        if "ʰ" in s:
            self.aspirated = True
            s = s.replace("ʰ", "")
        nfd = unicodedata.normalize("NFD", s)
        base = None
        for ch in nfd:
            cat = unicodedata.category(ch)
            if cat.startswith("L") and base is None:
                base = ch.lower()
            elif cat == "Mn":  # combining mark
                if ch == "̄":
                    self.long = True
                elif ch == "̃":
                    self.nasal = True
                elif ch == "̣":
                    self.retroflex = True
                # accents / dot-above / breve: ignored for features
            elif ch in ("ː",):
                self.long = True
        if base is None:
            return

        # precomposed retroflex letters already carry their own feature entry
        if raw in C:
            base = raw  # e.g. ʈ ɖ ṣ ṛ ḷ passed straight through
        elif self.retroflex and base in RETRO:
            base = RETRO[base]

        if base in V or base in "aeiouəɛεɔæɐʌʊɪ":
            self.kind = "V"
            self.height, self.back, rnd = V.get(base, ("mid", "central", 0))
            self.voice = True
        elif base in C:
            self.kind = "C"
            self.place, self.manner, self.voice = C[base]
            if self.retroflex:
                self.place = "retroflex"


def segments(tok, form: str) -> list[Seg]:
    return [Seg(g) for g in tok(form)]


# ─────────────────────────────────────────────────────────────────────────────
# Feature-weighted Needleman–Wunsch
# ─────────────────────────────────────────────────────────────────────────────

MANNER_CHAIN = ["stop", "affricate", "fricative", "approximant"]
GAP = -1.1


def _near(m, n):
    if m in MANNER_CHAIN and n in MANNER_CHAIN:
        return abs(MANNER_CHAIN.index(m) - MANNER_CHAIN.index(n)) == 1
    return False


def score(a: Seg, b: Seg) -> float:
    if a.raw == b.raw:
        return 2.4
    if a.kind == "V" and b.kind == "V":
        s = 0.8
        if a.height == b.height:
            s += 0.6
        if a.back == b.back:
            s += 0.6
        return s
    if a.kind == "C" and b.kind == "C":
        s = 0.0
        if a.place == b.place:
            s += 1.0
        if a.manner == b.manner:
            s += 1.2
        if a.voice == b.voice:
            s += 0.6
        if a.manner != b.manner and _near(a.manner, b.manner):
            s += 0.5
        return s - 0.4
    return -1.6


def align(e: list[Seg], r: list[Seg]):
    n, m = len(e), len(r)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * GAP
    for j in range(1, m + 1):
        dp[0][j] = j * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(dp[i - 1][j - 1] + score(e[i - 1], r[j - 1]),
                           dp[i - 1][j] + GAP, dp[i][j - 1] + GAP)
    # traceback by choosing the best predecessor (avoid fragile float == comparisons)
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        diag = dp[i - 1][j - 1] + score(e[i - 1], r[j - 1])
        up = dp[i - 1][j] + GAP
        left = dp[i][j - 1] + GAP
        if diag >= up and diag >= left:
            pairs.append((e[i - 1], r[j - 1])); i -= 1; j -= 1
        elif up >= left:
            pairs.append((e[i - 1], None)); i -= 1
        else:
            pairs.append((None, r[j - 1])); j -= 1
    while i > 0:
        pairs.append((e[i - 1], None)); i -= 1
    while j > 0:
        pairs.append((None, r[j - 1])); j -= 1
    pairs.reverse()
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Change labelling
# ─────────────────────────────────────────────────────────────────────────────

def manner_label(f, t):
    return {
        ("affricate", "stop"): "deaffrication", ("stop", "affricate"): "affrication",
        ("affricate", "fricative"): "spirantization", ("stop", "fricative"): "lenition",
        ("stop", "approximant"): "lenition", ("fricative", "stop"): "fortition",
        ("approximant", "stop"): "fortition",
    }.get((f, t))


def describe(e: Seg | None, r: Seg | None) -> str:
    """Return a compact change-category CODE. Surface labels are composed at render time from
    (etymon_seg, reflex_seg, code); the code is the linguistic categorisation and lives in the data.
    Codes: kept, loss, add | (V) nasalization, denasalization, lengthening, shortening, vowel |
    (C) nasalization, devoicing, voicing, deaffrication, affrication, spirantization, lenition,
    fortition, retroflexion, fronting, place, aspiration, deaspiration, cons."""
    if e and not r:
        return "loss"
    if r and not e:
        return "add"
    if e.raw == r.raw:
        return "kept"
    if e.kind == "V" and r.kind == "V":
        if not e.nasal and r.nasal:
            return "nasalization"
        if e.nasal and not r.nasal:
            return "denasalization"
        if not e.long and r.long:
            return "lengthening"
        if e.long and not r.long:
            return "shortening"
        return "vowel"
    if e.kind == "C" and r.kind == "C":
        if r.manner == "nasal" and e.manner != "nasal":
            return "nasalization"
        if e.voice and not r.voice:
            return "devoicing"
        if not e.voice and r.voice:
            return "voicing"
        ml = manner_label(e.manner, r.manner)
        if ml:
            return ml
        if e.place != r.place:
            return ("retroflexion" if r.place == "retroflex"
                    else "fronting" if e.place == "palatal" and r.place in ("dental", "alveolar")
                    else "place")
        if not e.aspirated and r.aspirated:
            return "aspiration"
        if e.aspirated and not r.aspirated:
            return "deaspiration"
        return "cons"
    return "vowel" if (e.kind == "V" or r.kind == "V") else "cons"


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="ALL", help="ALL, or a proto Language_ID e.g. Indo-Aryan")
    ap.add_argument("--limit", type=int, default=0, help="cap #origins (0 = all)")
    ap.add_argument("--out", default="cldf/alignments.csv")
    args = ap.parse_args()

    tok = segmenter(load_graphemes(PROFILE))

    with open(FORMS, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "Origin_ID" not in (reader.fieldnames or ()):
            raise ValueError("align.py must run after unify_cldf.py")
        rows = list(reader)
    etymon = {
        row["ID"]: (row["Form"], row["Language_ID"])
        for row in rows
    }

    keep_family = None if args.family == "ALL" else args.family
    seen_origins: set[str] = set()
    n_forms = n_pairs = 0

    with open(args.out, "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["Form_ID", "Origin_ID", "Pos", "Etymon_Idx",
                    "Etymon_Seg", "Reflex_Seg", "Change", "Prev_Seg", "Next_Seg"])
        ety_cache: dict[str, list[Seg]] = {}
        for row in rows:
            origin = row["Origin_ID"]
            meta = etymon.get(origin)
            if not meta:
                continue
            ename, efam = meta
            if keep_family and efam != keep_family:
                continue
            if efam not in PROTO_LANGS:
                continue  # only align where the parameter head is a real proto-form
            if args.limit and origin not in seen_origins and len(seen_origins) >= args.limit:
                continue
            seen_origins.add(origin)
            form = row["Form"]
            if not form or not ename:
                continue
            if origin not in ety_cache:
                es0 = segments(tok, ename)
                for k, s in enumerate(es0):
                    s.idx = k
                ety_cache[origin] = es0
            es = ety_cache[origin]
            rs = segments(tok, form)
            if not es or not rs:
                continue
            for pos, (a, b) in enumerate(align(es, rs)):
                if a is not None:
                    prev = es[a.idx - 1].raw if a.idx > 0 else "#"
                    nxt = es[a.idx + 1].raw if a.idx + 1 < len(es) else "#"
                else:
                    prev = nxt = ""  # insertion — no etymon context
                w.writerow([row["ID"], origin, pos, a.idx if a else -1,
                            a.raw if a else "", b.raw if b else "", describe(a, b), prev, nxt])
                n_pairs += 1
            n_forms += 1
            if n_forms % 20000 == 0:
                print(f"  …{n_forms} forms, {n_pairs} aligned segments", file=sys.stderr)

    print(f"[align] wrote {args.out}: {n_forms} reflexes, {n_pairs} aligned segments, "
          f"{len(seen_origins)} origins", file=sys.stderr)


if __name__ == "__main__":
    main()
