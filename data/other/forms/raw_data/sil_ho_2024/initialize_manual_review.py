#!/usr/bin/env python3
"""Create the immutable cell-by-cell visual-review ledger once.

The OCR text is copied only into a clearly labelled evidence column.  It never
populates the manual transcription, status, or acceptance columns.  This
initializer refuses to overwrite an existing ledger so completed review work
cannot be lost accidentally.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCAFFOLD = HERE / "ocr_scaffold.tsv"
OUTPUT = HERE / "manual_review.tsv"

FIELDS = [
    "Item",
    "Gloss",
    "Site_Code",
    "PDF_Page",
    "Printed_Page",
    "Column",
    "OCR_Evidence_Only",
    "Manual_Transcription",
    "Review_Status",
    "Confidence",
    "Uncertainty",
    "Reviewer_Method",
    "Reviewed_At",
]


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing ledger: {OUTPUT}")
    with SCAFFOLD.open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream, delimiter="\t"))
    if len(source) != 5_670:
        raise AssertionError("source topology drift")
    rows = []
    for row in source:
        rows.append({
            "Item": row["Item"],
            "Gloss": "",
            "Site_Code": row["Site_Code"],
            "PDF_Page": row["PDF_Page"],
            "Printed_Page": row["Printed_Page"],
            "Column": row["Column"],
            "OCR_Evidence_Only": row["OCR_Candidate"],
            "Manual_Transcription": "",
            "Review_Status": "unreviewed",
            "Confidence": "",
            "Uncertainty": "",
            "Reviewer_Method": "",
            "Reviewed_At": "",
        })
    with OUTPUT.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"initialized {len(rows)} unreviewed cells in {OUTPUT}")


if __name__ == "__main__":
    main()
