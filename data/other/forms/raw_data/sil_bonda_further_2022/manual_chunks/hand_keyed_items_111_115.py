#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 111-115.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_111_115_hand_keyed.tsv"
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
    (111, "son", (
        ("õʔõ", "6", ""), ("õʔõ", "6", ""),
        ("ʊʔʊ", "6", ""), ("ɔʔo", "6", ""),
        ("ɔʔo", "6", ""), ("ɔʔo", "6", ""),
        ("oɖuvon", "2", ""), ("hũŋ", "1", ""),
        ("ʌŋon", "3", ""), ("pɪlɛ", "4", ""),
        ("pu:o", "5", ""),
    )),
    (112, "daughter", (
        ("õʔõ", "7", ""), ("õʔõ", "7", ""),
        ("ʊʔʊ", "7", ""), ("ɔʔo", "7", ""),
        ("ɔʔo", "7", ""), ("selane", "6", ""),
        ("onoʔon", "2", ""),
        ("selamboinehũ | selamboinehũ", "1|8", "same response printed on separate group-1 and group-8 lines for BIA"),
        ("koɖon", "3", ""), ("tɔkɪ", "4", ""),
        ("dʒi:o", "5", ""),
    )),
    (113, "husband", (
        ("mpor", "6", ""), ("mpor", "6", ""),
        ("mpor", "6", ""), ("m̩por", "6", ""),
        ("m̩por", "6", ""), ("m̩por", "6", ""),
        ("remol", "2", ""), ("nehanɖa", "1", ""),
        ("mʌrɛʔ", "3", ""), ("mʊnʊs", "4", ""),
        ("suɑmi", "5", ""),
    )),
    (114, "wife", (
        ("kʊɳɪ", "6", ""), ("kʊɳuɪ", "6", ""),
        ("kʊɳuɪ", "6", ""), ("kʊni", "6", ""),
        ("kʊni", "6", ""), ("kʊni", "6", ""),
        ("kimoi | kʊmboi", "2|2", "two responses printed on separate group-2 lines for GUT"),
        ("nekəɳəi", "1", ""), ("kuɖumɛʔ", "3", ""),
        ("mɛjdʒɪ", "4", ""), ("st̪ri", "5", ""),
    )),
    (115, "boy", (
        ("ŋgere", "1", ""), ("ŋgere", "1", ""),
        ("ŋgere", "1", ""), ("gulene", "7", ""),
        ("gulene", "7", ""), ("ogu", "6", ""),
        ("oɖuvon", "2", ""), ("ŋgirboʔo", "1", ""),
        ("bubon", "3", ""), ("pɪlɛ", "4", ""),
        ("pilɑʔ | pu:o | pilɑʔ", "4|5|5", "group-4 response plus two comma-separated responses printed on the group-5 line for ODI"),
    )),
]


def source_page(item: int, site_code: str) -> tuple[str, str]:
    if item == 111 or (item == 112 and site_code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS"}):
        return "31", "26"
    return "32", "27"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 59
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
