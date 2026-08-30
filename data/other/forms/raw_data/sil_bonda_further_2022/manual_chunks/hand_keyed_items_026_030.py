#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 26-30.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_026_030_hand_keyed.tsv"
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

# Each cell is (manual transcription, similarity group, qualification).
PAGE_DECISIONS = [
    (26, "house", (
        ("d̪ijo", "5", ""), ("d̪ijo", "5", ""), ("d̪iŋõ", "5", ""),
        ("d̪ijo", "5", ""), ("d̪ijo", "5", ""), ("d̪ijo", "5", ""),
        ("d̪iɛn", "2", ""), ("d̪ʊa", "1", ""), ("ʌsuŋ", "4", ""),
        ("gəɾ", "3", ""), ("gɦoɾo", "3", ""),
    )),
    (27, "roof", (
        ("gʊd̪aŋbile", "7", ""), ("gʊd̪aŋbile", "7", ""),
        ("giraŋbulei", "7", ""), ("bɪleŋ", "6", ""),
        ("gʊd̪aŋbale", "7", ""), ("tʃʌuɳi", "4", ""),
        ("bɪlei", "2", ""), ("blesaŋ", "1", ""),
        ("bʌlɪŋsuŋ", "5", ""), ("tʃɛɳɪ", "4", ""),
        ("tʃha:to", "3", ""),
    )),
    (28, "door", (
        ("pʌt̪a", "3", ""), ("pʌt̪a", "3", ""), ("pʌt̪a", "3", ""),
        ("kəpat̪h", "1", ""), ("kəpat̪", "1", ""), ("kəpa", "1", ""),
        ("kəpet̪", "1", ""), ("kəpat̪", "1", ""), ("nʌŋenu", "2", ""),
        ("kɛpet̪", "1", ""), ("kəbato", "1", ""),
    )),
    (29, "firewood", (
        ("sʊŋo", "2", ""), ("sʊŋo", "2", ""), ("sʊŋo", "2", ""),
        ("suŋɔ", "2", ""), ("suŋɔ", "2", ""), ("suŋɔ", "2", ""),
        ("sʊø", "2", ""), ("sʊa", "2", ""), ("ʌŋʌl", "4", ""),
        ("d̪eɾʊ", "5", ""), ("ka:to", "3", ""),
    )),
    (30, "broom", (
        ("sʊnʊ", "2", ""), ("sʊnʊʔ", "2", ""), ("sʊnʊʔ", "2", ""),
        ("sunu", "2", ""), ("sunu", "2", ""), ("sunu", "2", ""),
        ("sʊnoq", "2", ""), ("tʃʊnɔ", "1", ""), ("dʒɔnɔ", "1", ""),
        ("bed̪ɳɪ", "4", ""), ("tʃantʃoni", "3", ""),
    )),
]


def source_pages(item: int, site_code: str) -> tuple[str, str]:
    if item <= 27 or (item == 28 and site_code in {"POD", "BON", "DUM", "KAD", "KEN"}):
        return "18", "13"
    return "19", "14"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 55
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
