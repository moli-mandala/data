#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 71-75.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_071_075_hand_keyed.tsv"
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
    (71, "rice", (
        ("ŋku", "4", ""), ("ŋku", "4", ""),
        ("ruŋkʊ", "1", ""), ("ŋku", "4", ""),
        ("ŋku", "4", ""), ("ŋku", "4", ""),
        ("rʊk:u", "1", ""), ("rʊk:o", "1", ""),
        ("ruŋ", "2", ""), ("tʃaul", "3", ""),
        ("tʃawulo", "3", ""),
    )),
    (72, "potato", (
        ("alu", "1", ""), ("alu", "1", ""),
        ("alu", "1", ""), ("alu", "1", ""),
        ("alu", "1", ""), ("alu", "1", ""),
        ("ɛlu", "1", ""), ("alu", "1", ""),
        ("ɛlu", "1", ""), ("ɛlʊ", "1", ""),
        ("aɭu", "1", ""),
    )),
    (73, "eggplant", (
        ("nijom", "2", ""), ("ɲiom", "2", ""),
        ("ɲiom", "2", ""), ("ijom", "2", ""),
        ("ijom", "2", ""), ("ijom", "2", ""),
        ("ejom | beigon", "2|3", "two responses printed on separate group-2 and group-3 lines for GUT"),
        ("koɖẽhẽ", "1", ""), ("bʌɪŋgɔn", "3", ""),
        ("bɛjgɔn", "3", ""), ("baiŋgonõ", "3", ""),
    )),
    (74, "groundnut", (
        ("sɔnʌ", "2", ""), ("sena", "2", ""),
        ("sena", "2", ""),
        ("tʃʌnɛ | tʃʌnɛ", "2|4", "same response printed on separate group-2 and group-4 lines for KAD"),
        ("tʃʌnɛ | tʃʌnɛ", "2|4", "same response printed on separate group-2 and group-4 lines for KEN"),
        ("buisenɛ", "2", ""), ("tʃen:ɛ", "2", ""),
        ("tʃertʃna", "4", ""), ("senɛ", "2", ""),
        ("tʃɛnɛ | tʃɛnɛ", "2|4", "same response printed on separate group-2 and group-4 lines for RON"),
        ("tʃinboɖam | tʃinboɖam", "1|3", "same response printed on separate group-1 and group-3 lines for ODI"),
    )),
    (75, "chili", (
        ("morsi", "2", ""), ("morsi", "2", ""),
        ("morsi", "2", ""), ("morsiŋ", "2", ""),
        ("morsiŋ", "2", ""), ("morsiŋ", "2", ""),
        ("morsi", "2", ""), ("miriŋ", "1", ""),
        ("mersɛ", "2", ""), ("mɔritʃ", "3", ""),
        ("mɔritʃə", "3", ""),
    )),
]


def source_pages(item: int, site_code: str) -> tuple[str, str]:
    if item <= 73 or (item == 74 and site_code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA"}):
        return "25", "20"
    return "26", "21"


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    pdf_page, printed_page = source_pages(item, site_code)
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": pdf_page,
        "Printed_Page": printed_page, "Column": column,
        "Manual_Transcription": form, "Similarity_Groups": groups,
        "Source_Qualification": qualification, "Review_Status": "attested",
        "Confidence": "high", "Uncertainty": "",
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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 60
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
