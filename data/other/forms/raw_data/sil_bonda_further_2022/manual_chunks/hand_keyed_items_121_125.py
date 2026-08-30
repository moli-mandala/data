#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 121-125.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_121_125_hand_keyed.tsv"
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
    (121, "evening/afternoon", (
        ("minɖip", "6", ""), ("minɖip", "6", ""),
        ("mʊnɖip", "6", ""), ("minɖi", "6", ""),
        ("minɖi", "6", ""), ("minɖi", "6", ""),
        ("nonɖigu", "2", ""), ("ləmɖig˥", "1", ""),
        ("uɖumɛ", "3", ""), ("sɔndʒbela", "4", ""),
        ("sənɖhija", "5", ""),
    )),
    (122, "yesterday", (
        ("t̪ʊgola", "1", ""), ("t̪ʊgola", "1", ""),
        ("t̪ʊgola", "1", ""), ("t̪ʊgo", "1", ""),
        ("t̪ʊgo", "1", ""), ("t̪ʊgo", "1", ""),
        ("mɛrt̪o", "2", ""), ("t̪ʊgʊa", "1", ""),
        ("uɖubun", "3", ""), ("kɛlɪ", "4", ""),
        ("kɑli", "4", ""),
    )),
    (123, "today", (
        ("oʔika", "5", ""), ("oʔika", "5", ""),
        ("aʔna", "6", ""), ("ɔiʔka", "5", ""),
        ("ɔiʔka", "5", ""), ("ɔiʔka", "5", ""),
        ("ʊsoŋ", "2", ""),
        ("eiʔke | eiʔke", "1|5", "same response printed on separate group-1 and group-5 lines for BIA"),
        ("miʈen", "3", ""), ("ɛdʒɪ", "4", ""),
        ("adʒi", "4", ""),
    )),
    (124, "tomorrow", (
        ("jeroga", "5", ""), ("jeroga", "5", ""),
        ("bera", "7", ""), ("jeroʔga", "5", ""),
        ("jeroʔga", "5", ""), ("jeroʔga", "5", ""),
        ("bier", "2", ""), ("mɖʒoɖe", "1", ""),
        ("bijog", "3", ""), ("kɛlɪ", "4", ""),
        ("asont̪akali", "4", ""),
    )),
    (125, "week", (
        ("at̪ek", "4", ""), ("at̪ek", "4", ""),
        ("at̪ek", "4", ""), ("ɛɖɪn", "2", ""),
        ("muinsan̪t̪a", "1", ""), ("at̪ek", "4", ""),
        ("ɛt̪ɖɪn", "2", ""), ("muisan̪t̪a", "1", ""),
        ("ɛt̪ɖɪn", "2", ""), ("ɛt̪ɖɪn", "2", ""),
        ("səpt̪ahə", "3", ""),
    )),
]


def source_page(item: int, site_code: str) -> tuple[str, str]:
    if item <= 124 or site_code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT", "BIA"}:
        return "33", "28"
    return "34", "29"


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    pdf_page, printed_page = source_page(item, site_code)
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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 56
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
