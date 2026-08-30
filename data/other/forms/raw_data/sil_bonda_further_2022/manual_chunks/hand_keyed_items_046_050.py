#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 46-50.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_046_050_hand_keyed.tsv"
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
    (46, "water", (
        ("d̪a", "2", ""), ("d̪aʔ", "2", ""), ("d̪aʔ", "2", ""),
        ("d̪ɛ", "2", ""), ("d̪ɛ", "2", ""), ("d̪ɛ", "2", ""),
        ("d̪ɛ", "2", ""), ("n̪d̪ia", "1", ""), ("d̪a", "4", ""),
        ("pəni", "3", ""), ("pəni", "3", ""),
    )),
    (47, "river", (
        ("kɪn̪d̪a", "3", ""), ("kɪn̪d̪aʔ", "3", ""),
        ("kɪn̪d̪aʔ", "3", ""), ("kɪn̪d̪ɛ", "3", ""),
        ("kɪn̪d̪ɛ", "3", ""), ("kɪn̪d̪ɛ", "3", ""),
        ("kɪn̪d̪i", "3", ""), ("kɪn̪d̪iɛ", "3", ""),
        ("kɪn̪d̪ɛ", "3", ""), ("gɛɾ", "4", ""),
        ("nəd̪i", "2", ""),
    )),
    (48, "cloud", (
        ("ɲuɾd̪a", "8", ""), ("ɲuɾd̪a", "8", ""),
        ("d̪aʔɲuɾgʊt̪a", "8", ""), ("bʌd̪ol", "4", ""),
        ("bʌd̪ol", "4", ""), ("bʌd̪ol", "4", ""),
        ("t̪ɪɾip", "2", ""), ("t̪ʊlet̪halodia", "6", ""),
        ("t̪eɾu", "5", ""), ("bed̪əl", "4", ""),
        ("megɦ:o", "3", ""),
    )),
    (49, "lightning", (
        ("ʊŋleid̪a | sɪn̪t̪aɾ", "6|7", "two responses printed on separate lines for POD"),
        ("ʊŋleid̪a", "6", ""), ("ʊŋleid̪a", "6", ""),
        ("bɪdʒɪli", "4", ""), ("bɪdʒɪli", "4", ""),
        ("bɪdʒɪli", "4", ""),
        ("moglei | dʒɪtki", "2|3", "two responses printed on separate lines for the same two-line GUT label"),
        ("lɔŋst̪aɾ", "1", ""), ("bɪdʒɪlɛ", "4", ""),
        ("bɪdʒɪlɪ", "4", ""), ("bidʒuli", "4", ""),
    )),
    (50, "rainbow", (
        ("gʊt̪ʊbʊɪ", "4", ""), ("gʊt̪ʊbu", "4", ""),
        ("oŋt̪ɪbu", "4", ""), ("gʊt̪ʊbu", "4", ""),
        ("gʊt̪ʊbu", "4", ""), ("gʊt̪ʊbu", "4", ""),
        ("iŋlet̪i", "2", ""), ("haʊe", "5", ""),
        ("bɪŋlet̪ɪ", "2", ""), ("in̪d̪ɔɾd̪ʊn̪ʊ", "3", ""),
        ("ind̪ɾod̪ənəsə", "3", ""),
    )),
]


def source_pages(item: int, site_code: str) -> tuple[str, str]:
    if item <= 46 or (item == 47 and site_code not in {"RON", "ODI"}):
        return "21", "16"
    return "22", "17"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 57
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
