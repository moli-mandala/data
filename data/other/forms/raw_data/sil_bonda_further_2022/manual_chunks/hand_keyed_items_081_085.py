#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 81-85.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_081_085_hand_keyed.tsv"
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
    (81, "cabbage", (
        ("pʊɖakobi", "1", ""), ("pʊɖakobi", "1", ""),
        ("ʊlakubi", "1", ""), ("kʊbi", "1", ""),
        ("pʊɖekʊbi", "1", ""), ("pʊɖekʊbi", "1", ""),
        ("pʊɖekobi | bənɖegobi", "1|2", "two responses printed on separate group-1 and group-2 lines for GUT"),
        ("puɖakobi", "1", ""), ("pɔʈɔrkobi", "3", ""),
        ("pʊrɛkɔbi", "1", ""), ("bənɖhakobi", "2", ""),
    )),
    (82, "oil", (
        ("sʊʔu", "6", ""), ("sʊʔu", "6", ""),
        ("suʔu", "6", ""), ("suʔu", "6", ""),
        ("suʔu", "6", ""), ("suʔu", "6", ""),
        ("soø:l", "2", ""), ("n̩tʃu", "1", ""),
        ("loruŋ", "3", ""), ("tʃɪkɔn", "4", ""),
        ("t̪elo", "5", ""),
    )),
    (83, "salt", (
        ("bɪʈɪ", "1", ""), ("bɪʈɪ", "1", ""),
        ("bʊʈɪ", "1", ""), ("bɪʈɪ", "1", ""),
        ("bɪʈɪ", "1", ""), ("bɪʈɪ", "1", ""),
        ("bɪʈɪ", "1", ""), ("bɪʈɪg", "1", ""),
        ("posu", "2", ""), ("nʊn", "4", ""),
        ("luŋə | nũno", "3|4", "two responses printed on separate group-3 and group-4 lines for ODI"),
    )),
    (84, "meat", (
        ("sili", "1", ""), ("seli", "1", ""),
        ("seli", "1", ""), ("seli", "1", ""),
        ("seli", "1", ""), ("seli", "1", ""),
        ("sel:i", "1", ""), ("tʃili", "1", ""),
        ("ɕiɕi", "2", ""), ("mɛus", "3", ""),
        ("maŋtso", "4", ""),
    )),
    (85, "fat", (
        ("kiri", "1", ""), ("kiri", "1", ""),
        ("kiri", "1", ""), ("kiri", "1", ""),
        ("kiri", "1", ""), ("kiri", "1", ""),
        ("kiri", "1", ""), ("kri", "1", ""),
        ("bɔs", "2", ""), ("bɔs", "2", ""),
        ("tʃərbi", "3", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "27", "Printed_Page": "22",
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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 57
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
