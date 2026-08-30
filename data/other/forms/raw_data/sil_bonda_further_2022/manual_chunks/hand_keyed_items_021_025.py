#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 21-25.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_021_025_hand_keyed.tsv"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Similarity_Groups",
    "Source_Qualification", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
SITES = [
    ("POD", "Podeiguda U. Bonda", "1"),
    ("BON", "Bondapada U. Bonda", "2"),
    ("DUM", "Dumripada U. Bonda", "3"),
    ("KAD", "Kadamguda L. Bonda", "4"),
    ("KEN", "Kendhuguda L. Bonda", "5"),
    ("RAS", "Rasabeda L. Bonda", "6"),
    ("GUT", "Tikrapada Gadaba", "7"),
    ("BIA", "Biapada U. Didayi", "8"),
    ("PAR", "Kinumun Parenga Parja", "9"),
    ("RON", "Malenga Rona Desiya", "10"),
    ("ODI", "Cuttack Oriya", "11"),
]

# Each attested cell is (manual transcription, similarity groups, qualification).
PAGE_DECISIONS = [
    (21, "heart", (
        ("dʒibon", "1", ""), ("dʒibon", "1", ""),
        ("dʒibon", "1", ""), ("dʒimon", "1", ""),
        ("gɪɾɛ", "6", ""),
        ("dʒimon | d̪ʊkd̪ukɪ", "1|5", "two responses printed on separate lines for RAS"),
        ("buq", "2", ""), ("dʒibon", "1", ""),
        ("buk", "2", ""), ("mən", "4", ""),
        ("həɾud̪aio", "3", ""),
    )),
    (22, "blood", (
        ("boni", "4", ""), ("boni", "4", ""), ("boni", "4", ""),
        ("bəni", "4", ""), ("bəni", "4", ""), ("bəni", "4", ""),
        ("jem", "2", ""), ("miã", "1", ""), ("mɪjəŋ", "1", ""),
        ("bəni", "4", ""), ("ɾakto", "3", ""),
    )),
    (25, "village", (
        ("ʊŋgəm", "2", ""), ("ʊŋgəm", "2", ""), ("ŋgom", "2", ""),
        ("uŋgəm", "2", ""), ("uŋgəm", "2", ""), ("uŋgəm", "2", ""),
        ("ʊŋgom", "2", ""), ("gʊɖa", "6", ""), ("bɪlɪŋ", "5", ""),
        ("gẽ", "4", ""), ("gɾa:mõ", "3", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str, status: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "18", "Printed_Page": "13",
        "Column": column, "Manual_Transcription": form,
        "Similarity_Groups": groups, "Source_Qualification": qualification,
        "Review_Status": status, "Confidence": "high", "Uncertainty": "",
        "Reviewer_Method": "manual visual inspection at 600 dpi; every cell rechecked in targeted 1200-dpi crops",
        "Reviewed_At": "2026-08-28", "Reviewer_Declaration": DECLARATION,
    }
    assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
    return row


def rows() -> list[dict[str, str]]:
    attested_by_item = {item: (gloss, cells) for item, gloss, cells in PAGE_DECISIONS}
    out: list[dict[str, str]] = []
    for item, gloss in ((21, "heart"), (22, "blood"), (23, "urine"),
                        (24, "feces"), (25, "village")):
        if item in {23, 24}:
            out.extend(
                make_row(
                    item, gloss, site, "", "",
                    "prompt printed DISQUALIFIED across all eleven lists",
                    "excluded_disqualified",
                )
                for site in SITES
            )
            continue
        _, cells = attested_by_item[item]
        assert len(cells) == len(SITES) == 11
        for site, (form, groups, qualification) in zip(SITES, cells, strict=True):
            out.append(make_row(item, gloss, site, form, groups, qualification, "attested"))
    return out


def main() -> None:
    output_rows = rows()
    assert len(output_rows) == 55
    assert len({(row["Item"], row["Site_Code"]) for row in output_rows}) == 55
    assert sum(row["Review_Status"] == "attested" for row in output_rows) == 33
    assert sum(row["Review_Status"] == "excluded_disqualified" for row in output_rows) == 22
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows if row["Manual_Transcription"]) == 34
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
