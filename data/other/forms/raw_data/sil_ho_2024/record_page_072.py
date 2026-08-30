#!/usr/bin/env python3
"""Record the manual, image-by-image transcription of PDF page 72.

This module contains no OCR-derived acceptance logic.  Every value below was
read directly from the rendered page at 180 dpi and keyed by printed item and
site code.  Similarity-class numerals are preserved as printed evidence.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "manual_review.tsv"
DATE = "2026-08-28"

SITES = [
    "HO1", "HTH", "HKA", "HKE", "HCH", "HCU", "HSU", "HSA", "HJO",
    "HDH", "HBG", "HO2", "HRA", "HO3", "HOP", "HBA", "HNI", "BBG",
    "BMA", "BOP", "BRA", "BGH", "MU1", "MU2", "SA1", "SBA", "OCU",
]

# A value of None is a visually confirmed printed dash (no response).
PAGE = {
    1: {
        "gloss": "body",
        "forms": [
            "1 homo", "1 homo", "1 homo", "1 homo", "1 homo", "1 homo",
            "1 homo", "1 homo", "1 homo", "1 homo", "1 homo", "1 homo",
            "1 homo", None, "1 homo", "1 homo", "1 homo", "1 horᵉmo",
            "1 hoɖomo", "1 hoɖomo", "1 hʌrᵉmo", "1 hoɖomo", None,
            "1 horᵉmo", "1 hormo", "1 hʌrmo", "2 soriro",
        ],
    },
    2: {
        "gloss": "head",
        "forms": [
            "1 bo?o", "1 bo?o", "1 bo?o", "1 bo?o", "1 bo?o", "1 bo?o",
            "1 bo?o", "1 bo?o", "1 bo?o", "1 bo?o", "1 bo?o", "1 bo",
            "1 bo?o", "1 bo?", "1 bo?o", "1 bo?o", "1 bo?o", "1 bo?o",
            "1 boho", "1 bo?o", "1 bo?o", "1 bo?", "1 bo?",
            "1 bo, 2 mund", "1 bohok", "1 bɔhʌ?", "2 mʊnɖɔ",
        ],
    },
    3: {
        "gloss": "hair",
        "forms": [
            "1 bo?o bale", "1 bo?o bale", "1 bo?o bale", "1 bo?o bale",
            "1 bo?o bale", "1 bo?o bale", "1 bo?o bale", "1 bo?o bale",
            "1 bo?o bale", "1 bo?o bale", "1 bale?", "2 ub",
            "1 bo?o bale", "2 u?p", "2 bo? up", "1 bo?o bale", "2 bo? up",
            "2 ub?", "2 ub", "2 bo?o up", "2 bo?o u?b", "2 u?p",
            "2 u?b", "2 ub", "2 u?p", "2 u?p", "1 bala",
        ],
    },
}

# These cells contain typewriter glyphs whose Unicode identity was resolved
# from the report's own alphabet chart rather than silently flattened.
MEDIUM = {
    (1, "BBG"): "raised/glided medial vowel retained as superscript e",
    (1, "BMA"): "source retroflex voiced stop encoded as IPA ɖ",
    (1, "BOP"): "source retroflex voiced stop encoded as IPA ɖ",
    (1, "BRA"): "source open-back vowel and raised medial vowel retained",
    (1, "BGH"): "source retroflex voiced stop encoded as IPA ɖ",
    (1, "MU2"): "raised/glided medial vowel retained as superscript e",
    (1, "SBA"): "source open-back vowel encoded as ʌ",
    (2, "SBA"): "source open-back vowels encoded as ɔ and ʌ",
    (2, "OCU"): "source vowel/retroflex glyphs encoded as ʊ and ɖ",
}


def main() -> None:
    with LEDGER.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    expected = {(item, site) for item in PAGE for site in SITES}
    seen = set()
    for row in rows:
        key = (int(row["Item"]), row["Site_Code"])
        if int(row["PDF_Page"]) != 72 or key not in expected:
            continue
        item, site = key
        form = PAGE[item]["forms"][SITES.index(site)]
        uncertainty = MEDIUM.get(key, "")
        values = {
            "Gloss": PAGE[item]["gloss"],
            "Manual_Transcription": form or "",
            "Review_Status": "blank" if form is None else "attested",
            "Confidence": "medium" if uncertainty else "high",
            "Uncertainty": uncertainty,
            "Reviewer_Method": "manual-source-image; rendered-180dpi; OCR-not-accepted",
            "Reviewed_At": DATE,
        }
        if row["Review_Status"] == "unreviewed":
            row.update(values)
        else:
            for field, value in values.items():
                if row[field] != value:
                    raise AssertionError(f"review ledger conflict at item {item}/{site}: {field}")
        seen.add(key)
    if seen != expected:
        raise AssertionError(f"page topology drift: {len(seen)} of {len(expected)}")
    with LEDGER.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("recorded 81 manually reviewed cells for PDF page 72 (items 1-3)")


if __name__ == "__main__":
    main()
