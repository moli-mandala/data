#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 161-165.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_161_165_hand_keyed.tsv"
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
    (161, "eleven", (
        ("egart̪a", "2", ""), ("egart̪a", "2", ""),
        ("egart̪a", "2", ""), ("egɛrot̪a", "2", ""),
        ("egɛrot̪a", "2", ""), ("egɛro", "2", ""),
        ("egɛr", "2", ""), ("gɔmiŋ", "1", ""),
        ("egɛr", "2", ""), ("ɛgɛr", "2", ""),
        ("egaɾo", "2", ""),
    )),
    (162, "twelve", (
        ("bart̪a", "2", ""), ("bart̪a", "2", ""),
        ("bart̪a", "2", ""), ("bɛrot̪a", "2", ""),
        ("bɛrot̪a", "2", ""), ("bɛro", "2", ""),
        ("bɛro", "2", ""), ("gɔmbar", "1", ""),
        ("bɛr", "2", ""), ("bɛr", "2", ""),
        ("baɾo", "2", ""),
    )),
    (163, "twenty", (
        ("kʊdije", "2", ""), ("kʊdije", "2", ""),
        ("kodije", "2", ""), ("koɖet̪a", "2", ""),
        ("koɖet̪a", "2", ""), ("kuɖijet̪a", "2", ""),
        ("koɖe", "2", ""), ("sʌlga", "1", ""),
        ("koɖe", "2", ""), ("kɔre", "2", ""),
        ("koɾiˑe", "2", ""),
    )),
    (164, "one hundred", (
        ("panskuɖi", "3", ""), ("panskuɖi", "3", ""),
        ("panskoɖi", "3", ""), ("pantʃkoɖi", "3", ""),
        ("pantʃkoɖi", "3", ""), ("pantʃkuɖi", "3", ""),
        ("soje", "2", ""), ("malkuɖi", "1", ""),
        ("soje", "2", ""), ("pẽtʃkɔrɪ", "3", ""),
        ("ekosaho", "4", ""),
    )),
    (165, "who?", (
        ("dʒa", "1", ""), ("dʒa", "1", ""),
        ("dʒa", "1", ""), ("dʒa", "1", ""),
        ("dʒa", "1", ""), ("dʒa", "1", ""),
        ("lɛi", "2", ""), ("dʒanɖe", "1", ""),
        ("moɪ", "3", ""), ("kɔn̪t̪e", "4", ""),
        ("kie", "5", ""),
    )),
]


def source_page(item: int, site_code: str) -> tuple[str, str]:
    if item <= 163 or (item == 164 and site_code in {"POD", "BON", "DUM", "KAD", "KEN"}):
        return "39", "34"
    return "40", "35"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 55
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
