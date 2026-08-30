#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 56-60.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_056_060_hand_keyed.tsv"
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
    (56, "smoke", (
        ("mʊksɪŋ", "4", ""), ("mʊksɪŋ", "4", ""),
        ("mʊkʔsɪŋ", "4", ""), ("mokʔsɪŋ", "4", ""),
        ("mokʔsɪŋ", "4", ""), ("mokʔsɪŋ", "4", ""),
        ("mʊʔsoŋ", "1", ""), ("maʔso", "1", ""),
        ("bud̪ɪ", "3", ""), ("d̪ʊẽ", "2", ""),
        ("d̪ɦuɑ̃", "2", ""),
    )),
    (57, "ash", (
        ("ʊksoŋ", "1", ""), ("ʊksoŋ", "1", ""),
        ("ʊksoŋ", "1", ""), ("ʊkʔsoŋ", "1", ""),
        ("ʊkʔsoŋ", "1", ""), ("ʊkʔsoŋ", "1", ""),
        ("ʊksoŋ", "1", ""), ("ʊkso", "1", ""),
        ("bud̪ɪ", "3", ""), ("tʃɛɾ", "4", ""),
        ("pɑ̃usə", "2", ""),
    )),
    (58, "mud", (
        ("lod̪ɪ", "6", ""), ("lod̪ɪ", "6", ""),
        ("lod̪ɪ", "6", ""),
        ("kʌd̪ot̪ʊbu | kʌd̪ot̪ʊbu", "4|5", "same response printed on separate group-4 and group-5 lines for KAD"),
        ("kʌd̪ot̪ʊbu | kʌd̪ot̪ʊbu", "4|5", "same response printed on separate group-4 and group-5 lines for KEN"),
        ("kʌsat̪ʊbu | kʌsat̪ʊbu", "2|5", "same response printed on separate group-2 and group-5 lines for RAS"),
        ("t̪ʊbo", "2", ""), ("bʊɾd̪a", "1", ""),
        ("lobo", "3", ""), ("kad̪ɔ", "4", ""),
        ("kad̪uə", "4", ""),
    )),
    (59, "dust", (
        ("d̪ʊli", "2", ""), ("d̪ʊli", "2", ""),
        ("d̪ʊli", "2", ""), ("d̪ʊli", "2", ""),
        ("d̪ʊli", "2", ""), ("d̪ʊli", "2", ""),
        ("d̪ʊli", "2", ""), ("t̪hʊpʊɾlo", "1", ""),
        ("d̪ʊlɪ", "2", ""), ("d̪ʊlɪ", "2", ""),
        ("d̪ɦuɭi", "2", ""),
    )),
    (60, "gold", (
        ("suna", "1", ""), ("suna", "1", ""),
        ("suna", "1", ""), ("sʊnɛ", "1", ""),
        ("sʊnɛ", "1", ""), ("sʊnɛ", "1", ""),
        ("sʊn:ɛ", "1", ""), ("suna", "1", ""),
        ("sonɛ", "1", ""), ("sʊnɛ", "1", ""),
        ("sun:a", "1", ""),
    )),
]


def source_pages(item: int, site_code: str) -> tuple[str, str]:
    if item <= 59 or (item == 60 and site_code not in {"RON", "ODI"}):
        return "23", "18"
    return "24", "19"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 58
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
