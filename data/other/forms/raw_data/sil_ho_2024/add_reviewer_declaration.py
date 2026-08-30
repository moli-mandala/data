#!/usr/bin/env python3
"""Add the explicit hand-keying declaration to finalized base-ledger rows."""
from __future__ import annotations

import csv
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "manual_review.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"


def main() -> None:
    with LEDGER.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if "Reviewer_Declaration" not in fields:
        fields.append("Reviewer_Declaration")
    for row in rows:
        expected = "" if row["Review_Status"] == "unreviewed" else DECLARATION
        current = row.get("Reviewer_Declaration", "")
        if current not in {"", expected}:
            raise ValueError(f"Conflicting declaration for {row['Item']}+{row['Site_Code']}")
        row["Reviewer_Declaration"] = expected
    with LEDGER.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"declared {sum(bool(row['Reviewer_Declaration']) for row in rows)} finalized manual rows")


if __name__ == "__main__":
    main()
