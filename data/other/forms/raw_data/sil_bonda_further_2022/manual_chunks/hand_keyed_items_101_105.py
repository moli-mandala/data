#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 101-105.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_101_105_hand_keyed.tsv"
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

# Each cell is (manual transcription, similarity groups, qualification).
PAGE_DECISIONS = [
    (101, "name", (
        ("imi", "1", ""), ("imi", "1", ""),
        ("imi", "1", ""), ("əmi", "1", ""),
        ("əmi", "1", ""), ("əmi", "1", ""),
        ("imi | nev", "1|2", "two responses printed on separate group-1 and group-2 lines for GUT"),
        ("mɳi", "1", ""), ("nʌʊ", "2", ""),
        ("nã", "2", ""), ("nɑ:mə", "3", ""),
    )),
    (102, "man", (
        ("ŋgera | remo", "6|6", "two responses printed comma-separated on the group-6 line for POD"),
        ("ŋgere", "6", ""), ("ŋgere", "6", ""),
        ("ŋgere", "6", ""), ("ŋgere", "6", ""),
        ("ŋgere", "6", ""), ("oɖuvon", "2", ""),
        ("gɪrboi", "1", ""), ("ɛɖu", "3", ""),
        ("mʊnʊs", "4", ""), ("moniʃo", "5", ""),
    )),
    (103, "woman", (
        ("selane", "7", ""), ("selane", "7", ""),
        ("selane", "7", ""), ("selane", "7", ""),
        ("selane", "7", ""), ("selamboi", "1", ""),
        ("onoʔon", "2", ""), ("selamboi", "1", ""),
        ("guɳʈɔr | ʌmkur", "3|4", "two responses printed on separate group-3 and group-4 lines for PAR"),
        ("mɛjdʒɪ", "5", ""), ("st̪ri", "6", ""),
    )),
    (104, "child", (
        ("gulaine", "9", ""), ("gulaine", "9", ""),
        ("gʊbʊle", "8", ""), ("gu:", "6", ""),
        ("gu:", "6", ""), ("gu:", "6", ""),
        ("oø:n", "2", ""), ("tʃɛrlao", "1", ""),
        ("buboŋ", "3", ""), ("pɪlɛʈɔkɪ", "4", ""),
        ("pilɑ", "5", ""),
    )),
    (105, "father", (
        ("ba", "1", ""), ("ba", "1", ""),
        ("ba", "1", ""), ("ba:", "1", ""),
        ("ba:", "1", ""), ("ba:", "1", ""),
        ("əbeŋ", "1", ""), ("m̩ba", "1", ""),
        ("ʌbɛ | ʌbɛ", "1|2", "same response printed on separate group-1 and group-2 lines for PAR"),
        ("bɛbɛ", "2", ""), ("bapɑ", "2", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "30", "Printed_Page": "25",
        "Column": column, "Manual_Transcription": form,
        "Similarity_Groups": groups, "Source_Qualification": qualification,
        "Review_Status": "attested", "Confidence": "high", "Uncertainty": "",
        "Reviewer_Method": "manual visual inspection at 600 dpi; every cell rechecked in targeted 1200-dpi crops",
        "Reviewed_At": "2026-08-29", "Reviewer_Declaration": DECLARATION,
    }
    assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
    return row


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item, gloss, cells in PAGE_DECISIONS:
        assert len(cells) == len(SITES) == 11
        for site, (form, groups, qualification) in zip(SITES, cells, strict=True):
            out.append(make_row(item, gloss, site, form, groups, qualification))
    return out


def main() -> None:
    output_rows = rows()
    assert len(output_rows) == 55
    assert len({(row["Item"], row["Site_Code"]) for row in output_rows}) == 55
    assert all(row["Review_Status"] == "attested" for row in output_rows)
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 59
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
