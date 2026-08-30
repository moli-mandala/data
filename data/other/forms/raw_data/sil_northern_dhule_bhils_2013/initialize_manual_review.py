#!/usr/bin/env python3
"""Create the immutable 2,730-cell manual-review ledger once.

No OCR is read or copied by this initializer.  Every accepted response must
arrive later in an OCR-blind hand-keyed chunk and pass the guarded importer.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "manual_review.tsv"
SITES = "KEL DHA DIG AMO MUN AST MAN BHU AML SEG KAN SHA TOR".split()
FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Manual_Transcription", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]


def page_for(item: int) -> int:
    """Return the physically printed matrix page, including irregular leaves."""
    if item <= 45:
        return 91 + (item - 1) // 5
    if item <= 49:
        return 100
    if item <= 194:
        return 101 + (item - 50) // 5
    if item <= 199:
        return 130
    if item <= 204:
        return 131
    if item <= 209:
        return 132
    return 133


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing ledger: {OUTPUT}")
    rows = []
    for item in range(1, 211):
        page = page_for(item)
        for index, site in enumerate(SITES):
            rows.append({
                "Item": item,
                "Gloss": "",
                "Site_Code": site,
                "PDF_Page": page,
                "Printed_Page": page - 8,
                "Column": "left" if index < 6 else "right",
                "Manual_Transcription": "",
                "Review_Status": "unreviewed",
                "Confidence": "",
                "Uncertainty": "",
                "Reviewer_Method": "",
                "Reviewed_At": "",
                "Reviewer_Declaration": "",
            })
    with OUTPUT.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"initialized {len(rows)} unreviewed cells in {OUTPUT}")


if __name__ == "__main__":
    main()
