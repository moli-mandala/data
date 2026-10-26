#!/usr/bin/env python3
"""
extract.py — write the (etymon, reflex) eval set for one parent→child pair.

Streams cldf/forms.csv + cldf/edges.csv once and keeps only the two languages involved, so the
output TSV is small and every later eval/induction step reads just that file.

A child form is included when its rank-1 attestation chain (following `variant` edges upward)
reaches a parent-language form through a `reflex` edge. `borrowed` edges are excluded: loans are
not expected to obey the inherited sound changes.

The split column assigns whole etyma (never individual reflexes) to train/test by a stable hash,
so no etymon is seen in both.

Usage (from data/):
    python sound_changes/extract.py --parent Indo-Aryan --child Pk --out sound_changes/pairs/OIA-Pk.tsv
    python sound_changes/extract.py --all-min-etyma 100 --outdir sound_changes/pairs   # every child ≥100 etyma
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def split_of(etymon_id: str, test_pct: int) -> str:
    h = int(hashlib.sha1(etymon_id.encode()).hexdigest()[:8], 16)
    return "test" if h % 100 < test_pct else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="Indo-Aryan")
    ap.add_argument("--child", action="append", help="child Language_ID (repeatable)")
    ap.add_argument("--all-min-etyma", type=int, default=0,
                    help="instead of --child: every language with ≥N distinct linked etyma")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs"))
    ap.add_argument("--forms", default="cldf/forms.csv")
    ap.add_argument("--edges", default="cldf/edges.csv")
    ap.add_argument("--test-pct", type=int, default=20)
    ap.add_argument("--out", help="output path (single --child only)")
    args = ap.parse_args()

    lang_of: dict[str, str] = {}
    info: dict[str, tuple[str, str, str]] = {}
    with open(args.forms, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang_of[row["ID"]] = row["Language_ID"]
            info[row["ID"]] = (row["Form"], row["Gloss"], row["Source"].split(";")[0].split("[")[0])

    rank1: dict[str, tuple[str, str]] = {}
    with open(args.edges, encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e["Rank"] == "1" and e["Kind"] in ("reflex", "borrowed", "variant"):
                rank1[e["Child_ID"]] = (e["Parent_ID"], e["Kind"])

    def etymon_of(node: str) -> str | None:
        seen = set()
        cur = node
        while cur not in seen:
            seen.add(cur)
            e = rank1.get(cur)
            if e is None:
                return None
            parent, kind = e
            if kind == "borrowed":
                return None
            if kind == "reflex":
                return parent if lang_of.get(parent) == args.parent else None
            cur = parent  # variant: keep walking
        return None

    by_child: dict[str, list] = defaultdict(list)
    for fid, (form, gloss, src) in info.items():
        lang = lang_of[fid]
        if lang == args.parent or not form.strip() or "°" in form:
            continue  # `°` marks an abbreviated fragment (°ṇiyā, agu°), not a whole word
        if args.child and lang not in args.child:
            continue
        eid = etymon_of(fid)
        if not eid or not info[eid][0].strip():
            continue
        by_child[lang].append((eid, fid, form, gloss, src))

    if args.all_min_etyma:
        children = sorted(c for c, rows in by_child.items()
                          if len({r[0] for r in rows}) >= args.all_min_etyma)
    else:
        children = args.child or []
    all_ids = set(lang_of.values())
    for child in children:
        out = args.out if (args.out and len(children) == 1) else \
            os.path.join(args.outdir, f"OIA-{safe(child, all_ids)}.tsv")
        n = 0
        seen_pairs = set()
        with open(out, "w", newline="", encoding="utf-8") as g:
            w = csv.writer(g, delimiter="\t", lineterminator="\n")
            w.writerow(["Etymon_ID", "Etymon", "Reflex_ID", "Reflex", "Gloss", "Split", "Source"])
            for eid, fid, form, gloss, src in by_child[child]:
                if (eid, form) in seen_pairs:
                    continue
                seen_pairs.add((eid, form))
                w.writerow([eid, info[eid][0], fid, form, gloss or info[eid][1],
                            split_of(eid, args.test_pct), src])
                n += 1
        n_ety = len({r[0] for r in by_child[child]})
        print(f"{child}\t{n_ety}\t{n}\t{out}", file=sys.stderr)


def safe(lang_id: str, all_ids=()) -> str:
    """File stem for a Language_ID: IDs may contain '.', '/', spaces, and some differ only by case
    (As Ashokan vs as Domari), which collide on case-insensitive file systems — those get a
    case-pattern suffix (As → As-Ul, as → as-ll)."""
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in lang_id)
    if sum(1 for other in all_ids if other.lower() == lang_id.lower()) > 1:
        stem += "-" + "".join("U" if ch.isupper() else "l" for ch in lang_id if ch.isalpha())
    return stem


if __name__ == "__main__":
    main()
