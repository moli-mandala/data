#!/usr/bin/env python3
"""One-time audited correction of the unreviewed base-ledger page scaffold.

Physical pages 100, 130, 132, and 133 have irregular prompt counts. This
utility refuses any merged/reviewed ledger and changes coordinate fields only.
"""
from __future__ import annotations

import csv
from pathlib import Path

from initialize_manual_review import FIELDS, OUTPUT, page_for


def main() -> None:
    with OUTPUT.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
        if list(reader.fieldnames or ()) != FIELDS:
            raise SystemExit("unexpected base-ledger schema")
    if len(rows) != 2730 or any(row["Review_Status"] != "unreviewed" for row in rows):
        raise SystemExit("refusing to rewrite a non-pristine base ledger")
    changed = 0
    for row in rows:
        page = page_for(int(row["Item"]))
        if row["PDF_Page"] != str(page) or row["Printed_Page"] != str(page - 8):
            row["PDF_Page"] = str(page)
            row["Printed_Page"] = str(page - 8)
            changed += 1
    temporary = OUTPUT.with_suffix(".tsv.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(OUTPUT)
    print(f"corrected page coordinates in {changed} unreviewed base rows")


if __name__ == "__main__":
    main()
