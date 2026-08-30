#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 61-65.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_061_065_hand_keyed.tsv"
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
    (61, "tree", (
        ("semʊ", "5", ""), ("semʊ", "5", ""),
        ("sʊlop", "2", ""), ("çemu", "5", ""),
        ("çemu", "5", ""), ("çemu", "5", ""),
        ("sʊl:op", "2", ""), ("sɭa", "1", ""),
        ("ʌrɛ", "4", ""), ("gɔtʃ", "3", ""),
        ("gɔtʃhə", "3", ""),
    )),
    (62, "leaf", (
        ("ʊla", "4", ""), ("ʊla", "4", ""),
        ("ʊla", "4", ""), ("ʊla", "4", ""),
        ("ʊla", "4", ""), ("ʊla", "4", ""),
        ("ol:ɛ", "4", ""), ("ʊlija", "1", ""),
        ("ʊolɛ | ʊolɛ", "2|4", "same response printed on separate group-2 and group-4 lines for PAR"),
        ("pɔt̪ɔr", "3", ""), ("pɔt̪ɔr", "3", ""),
    )),
    (63, "root", (
        ("reɪgi", "4", ""), ("reʔgi", "4", ""),
        ("reʔgi", "4", ""), ("tʃɛr", "2", ""),
        ("tʃɛr", "2", ""), ("tʃɛr", "2", ""),
        ("sɛr", "2", ""), ("n̪dʒrɛ", "1", ""),
        ("ʃɛr", "2", ""), ("tʃɛr", "2", ""),
        ("tʃɛro", "2", ""),
    )),
    (64, "thorn", (
        ("gre", "1", ""), ("giraɪ", "2", ""),
        ("girʌi", "2", ""), ("girʌi", "2", ""),
        ("girʌi", "2", ""), ("gʌrʌi", "2", ""),
        ("gʊrɛi", "2", ""), ("grɛ", "1", ""),
        ("ube | ʊolɛ", "3|5", "two responses printed on separate group-3 and group-5 lines for PAR"),
        ("kẽt̪e", "4", ""), ("kon̪a", "6", ""),
    )),
    (65, "flower", (
        ("sʌri", "1", ""), ("sari", "1", ""),
        ("sari", "1", ""), ("sari", "1", ""),
        ("sari", "1", ""), ("sari", "1", ""),
        ("sɛri", "1", ""), ("sari", "1", ""),
        ("t̪ʌrbɛ", "2", ""), ("phʊl", "3", ""),
        ("phulo", "3", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "24", "Printed_Page": "19",
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
