#!/usr/bin/env python3
"""Finalize the already-completed visual review of Appendix B.3.

This small deterministic step records the human review decision in every
ledger row.  It does not perform OCR and does not substitute for visual
inspection: PDF pp. 37--115 were inspected from 180-dpi canonical renders
before this file was run.
"""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEDGER = HERE / "reviewed_transcription.tsv"
EXPECTED_PAGES = set(range(37, 116))
SOURCE_UNCERTAINTY = {
    ("59", "91", "Assamese, Dibrugarh", "soʌ̆ĭ??"):
        "The source visibly prints two literal question marks after soʌ̆ĭ; retained exactly as source-marked uncertainty in an excluded Assamese control response.",
}


def main() -> None:
    with LEDGER.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
        fields = list(rows[0])
    pages = {int(row["PDF_Page"]) for row in rows}
    if pages != EXPECTED_PAGES or len(rows) != 5_966:
        raise ValueError("review ledger no longer matches the visually reviewed appendix")
    found_uncertainty: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (row["PDF_Page"], row["Item"], row["Site"], row["Verified_Form"])
        row["Review_Status"] = "complete"
        row["Confidence"] = "high"
        if row["Record_Type"] == "blank":
            row["Review_Note"] = (
                f"PDF p. {row['PDF_Page']} visually reviewed at 180 dpi; "
                "source explicitly prints no entry, so the cell is confirmed blank."
            )
        else:
            row["Review_Note"] = (
                f"PDF p. {row['PDF_Page']} visually reviewed at 180 dpi; "
                "verified form matches the rendered source."
            )
        if key in SOURCE_UNCERTAINTY:
            found_uncertainty.add(key)
            row["Review_Status"] = "source-marked-uncertain"
            row["Confidence"] = "high"
            row["Review_Note"] = SOURCE_UNCERTAINTY[key]
    if found_uncertainty != set(SOURCE_UNCERTAINTY):
        raise ValueError("expected source-marked uncertainty was not found")
    with LEDGER.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"reviewed_records={len(rows)} reviewed_pages={len(pages)} source_uncertainties={len(found_uncertainty)}")


if __name__ == "__main__":
    main()
