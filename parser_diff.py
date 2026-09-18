#!/usr/bin/env python3
"""Show what a dictionary-parser change did, without a full CLDF build.

Compares the parser's CSV in the working tree against the committed one (``git show HEAD:``)
row by row, keyed on (entry, language, form), and prints counts of glosses and notes gained,
lost and changed plus samples. Run the parser first (``make cdial`` / ``make dedr``), or pass
``--run`` to do both in one step.

    uv run python parser_diff.py cdial
    uv run python parser_diff.py dedr --run --samples 40
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PARSERS = {
    # name: (csv path, gloss column, notes column, parser directory)
    "cdial": ("data/cdial/cdial.csv", 3, 6, "data/cdial"),
    "dedr": ("data/dedr/dedr_new.csv", 3, 6, "data/dedr"),
}


def rows_by_key(rows):
    """Index rows by (entry, language, form, occurrence) so insertions do not shift the diff."""
    seen = collections.Counter()
    out = {}
    for row in rows:
        key = (row[1], row[0], row[2])
        seen[key] += 1
        out[(*key, seen[key])] = row
    return out


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("parser", choices=sorted(PARSERS))
    cli.add_argument("--run", action="store_true", help="run the parser before diffing")
    cli.add_argument("--samples", type=int, default=20)
    cli.add_argument("--seed", type=int, default=0)
    args = cli.parse_args()

    path, gloss, notes, directory = PARSERS[args.parser]
    if args.run:
        subprocess.run([sys.executable, "parse.py"], cwd=ROOT / directory, check=True)

    csv.field_size_limit(10**9)
    committed = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    old = rows_by_key(csv.reader(io.StringIO(committed)))
    with (ROOT / path).open(encoding="utf-8", newline="") as stream:
        new = rows_by_key(csv.reader(stream))

    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    shared = sorted(set(new) & set(old))
    gloss_changes = [(k, old[k][gloss], new[k][gloss]) for k in shared if old[k][gloss] != new[k][gloss]]
    note_changes = [(k, old[k][notes], new[k][notes]) for k in shared if old[k][notes] != new[k][notes]]
    gained = [c for c in gloss_changes if not c[1] and c[2]]
    lost = [c for c in gloss_changes if c[1] and not c[2]]
    edited = [c for c in gloss_changes if c[1] and c[2]]
    blank_old = sum(1 for r in old.values() if not r[gloss].strip())
    blank_new = sum(1 for r in new.values() if not r[gloss].strip())

    print(f"{args.parser}: {len(old)} rows committed → {len(new)} in working tree "
          f"({len(added)} added, {len(removed)} removed)")
    print(f"glosses: {len(gained)} filled, {len(lost)} blanked, {len(edited)} edited; "
          f"blank {blank_old} → {blank_new}")
    print(f"notes:   {len(note_changes)} changed")

    random.seed(args.seed)

    def show(title, items, render):
        if not items:
            return
        print(f"\n== {title} ({len(items)}; showing up to {args.samples})")
        for item in random.sample(items, min(args.samples, len(items))):
            print("  " + render(item))

    show("glosses blanked", lost, lambda c: f"{c[0][0]:<7} {c[0][1]:<8} {c[0][2]:<18} {c[1][:50]!r} -> ''")
    show("glosses edited", edited, lambda c: f"{c[0][0]:<7} {c[0][1]:<8} {c[0][2]:<18} {c[1][:35]!r} -> {c[2][:35]!r}")
    show("glosses filled", gained, lambda c: f"{c[0][0]:<7} {c[0][1]:<8} {c[0][2]:<18} -> {c[2][:60]!r}")
    show("notes changed", note_changes, lambda c: f"{c[0][0]:<7} {c[0][1]:<8} {c[0][2]:<18} {c[1][:30]!r} -> {c[2][:30]!r}")
    show("rows removed", removed, lambda k: f"{k[0]:<7} {k[1]:<8} {k[2]:<18} {old[k][gloss][:50]!r}")
    show("rows added", added, lambda k: f"{k[0]:<7} {k[1]:<8} {k[2]:<18} {new[k][gloss][:50]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
