#!/usr/bin/env python3
"""One-off identity migration for etymon rows in ``data/other/params/*.csv``.

Column 5 of an ``other/params`` file is its citation string, but ``make_cldf.py``
dropped it, so Strand's Proto-Indo-Iranian and Persian etyma reached
``cldf/forms.csv`` unattributed.  Carrying it through is a one-line fix — but
``Source`` is part of the durable-ID fingerprint, so on the next build every one
of those rows fingerprints differently, fails to match its registry entry, and is
re-minted with a fresh ``f_`` id.  That silently breaks the curated
``data/etymology-assignments.csv`` rows that point at them.

This script updates the affected registry rows *in place* — same ``Form_ID``, same
``Legacy_ID`` — to the source and fingerprint the corrected build produces.  It
refuses to touch a row whose identity differs in any other field, and it is a
no-op once applied.

    uv run python migrate_params_source_identity.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

from assign_form_ids import REGISTRY, REGISTRY_FIELDS, fingerprint, read_rows, write_rows

ROOT = Path(__file__).resolve().parent


def params_sources() -> dict[str, str]:
    """Legacy param id → the citation string its source file states."""
    out: dict[str, str] = {}
    for path in sorted(glob.glob(str(ROOT / "data/other/params/*.csv"))):
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) > 4 and row[0] and row[4].strip():
                    out[row[0]] = row[4].strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forms", type=Path, default=ROOT / "cldf/forms.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = params_sources()
    csv.field_size_limit(10 ** 9)
    # Take the target fingerprint from the build itself rather than recomputing it
    # from the registry snapshot, so the migration cannot drift from assign_ids().
    target: dict[str, str] = {}
    with args.forms.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["ID"] in sources:
                target[row["ID"]] = fingerprint(row)

    fields, registry = read_rows(REGISTRY)
    changed, skipped = 0, []
    for row in registry:
        legacy = row.get("Legacy_ID", "")
        source = sources.get(legacy)
        if not source or row.get("Status") != "active" or legacy not in target:
            continue
        if row.get("Fingerprint") == target[legacy] and row.get("Source") == source:
            continue          # already migrated
        if row.get("Source", ""):
            skipped.append((legacy, row["Source"], source))
            continue          # a real source change is not this migration's business
        row["Source"] = source
        row["Fingerprint"] = target[legacy]
        changed += 1

    print(f"{changed:,} registry rows re-fingerprinted with their params citation")
    if skipped:
        print(f"{len(skipped):,} rows left alone (they already carry a different source)")
        for entry in skipped[:5]:
            print("   ", entry)
    if not args.dry_run and changed:
        write_rows(REGISTRY, REGISTRY_FIELDS, registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
