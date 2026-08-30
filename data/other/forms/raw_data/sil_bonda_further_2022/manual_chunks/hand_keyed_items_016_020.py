#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 16-20.

Every conceptual cell was independently reviewed from 600-dpi rendered pages
and rechecked in targeted 1200-dpi crops. OCR, PDF text, and prior-source
readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_016_020_hand_keyed.tsv"
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
    (16, "finger", (
        ("nd̪roit̪i", "4", ""), ("nd̪ruait̪i", "4", ""),
        ("nd̪ruait̪i", "4", ""), ("ʌŋti", "3", ""),
        ("ʌŋti", "3", ""), ("ʌŋti", "3", ""),
        ("əŋti", "3", ""), ("vʌɾvat̪i", "1", ""),
        ("ʌŋti", "3", ""), ("əŋti", "3", ""),
        ("aŋguɭi", "2", ""),
    )),
    (17, "fingernail", (
        ("kɪrime", "4", ""), ("kɪrime", "4", ""),
        ("kɪrime", "4", ""), ("kɪrime", "4", ""),
        ("kɪrime", "4", ""), ("kɪrime", "4", ""),
        ("ɾʊmɪ | noq", "2|3", "two responses printed on separate lines for the same two-line GUT label"),
        ("n̩tʃəit̪i", "1", ""), ("ɾuʊəɾ", "2", ""),
        ("nək", "3", ""), ("no:kho", "3", ""),
    )),
    (18, "leg", (
        ("t̪iksʊŋ", "6", ""), ("t̪iksʊŋ", "6", ""),
        ("t̪iksʊŋ", "6", ""), ("t̪eksuŋ", "6", ""),
        ("t̪eksuŋ", "6", ""), ("t̪esuŋ", "6", ""),
        ("sʊsʊŋ", "2", ""), ("ʊntʃə", "1", ""),
        ("dʒiŋ", "5", ""), ("gəɾ", "4", ""), ("gudo", "3", ""),
    )),
    (19, "skin", (
        ("ʊsa", "1", ""), ("ʊsa", "1", ""), ("ɪsa", "1", ""),
        ("ʊsa", "1", ""), ("ʊsa", "1", ""), ("ʊsa", "1", ""),
        ("ɪsel", "2", ""), ("ʊksa", "1", ""), ("sɪsɪ", "4", ""),
        ("tʃem", "5", ""), ("tʃaɾəmõ", "3", ""),
    )),
    (20, "bone", (
        ("sɪksəŋ", "2", ""), ("sɪksəŋ", "2", ""),
        ("sɪksəŋ", "2", ""), ("sɪksəŋ", "2", ""),
        ("sɪksəŋ", "2", ""), ("sɪksəŋ", "2", ""),
        ("sɪsəŋ", "2", ""), ("ntʃia", "1", ""),
        ("dʒɛ", "3", ""), ("əɾ", "5", ""), ("ha:do", "4", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    pdf_page, printed_page = ("17", "12") if item <= 19 else ("18", "13")
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": pdf_page,
        "Printed_Page": printed_page, "Column": column,
        "Manual_Transcription": form, "Similarity_Groups": groups,
        "Source_Qualification": qualification, "Review_Status": "attested",
        "Confidence": "high", "Uncertainty": "",
        "Reviewer_Method": "manual visual inspection at 600 dpi; every cell rechecked in targeted 1200-dpi crops",
        "Reviewed_At": "2026-08-28", "Reviewer_Declaration": DECLARATION,
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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 56
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
