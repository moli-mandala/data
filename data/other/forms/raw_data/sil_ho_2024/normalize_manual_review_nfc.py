#!/usr/bin/env python3
"""Normalize the manually reviewed ledger to canonical Unicode NFC in place."""
from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "manual_review.tsv"


def main() -> None:
    with LEDGER.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    assert fields
    changed = 0
    for row in rows:
        for field, value in row.items():
            normalized = unicodedata.normalize("NFC", value)
            if normalized != value:
                row[field] = normalized
                changed += 1
    with LEDGER.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"NFC-normalized manual_review.tsv ({changed} changed field values)")


if __name__ == "__main__":
    main()
