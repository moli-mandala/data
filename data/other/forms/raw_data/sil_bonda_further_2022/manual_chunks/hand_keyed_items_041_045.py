#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 41-45.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_041_045_hand_keyed.tsv"
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
    (41, "sun", (
        ("siŋi", "1", ""), ("siŋi", "1", ""), ("siŋi", "1", ""),
        ("sĩ", "2", ""), ("sĩ", "2", ""), ("sĩ", "2", ""),
        ("siø", "2", ""), ("çini", "1", ""), ("enki", "5", ""),
        ("bel", "4", ""), ("sudʒo", "3", ""),
    )),
    (42, "moon", (
        ("ʌɾke", "1", ""), ("ʌɾke", "1", ""), ("ʌɾke", "1", ""),
        ("ʌɾke", "1", ""), ("ʌɾke", "1", ""), ("ʌɾke", "1", ""),
        ("eɾkɛ", "1", ""), ("ʌɾke", "1", ""), ("ʌŋʌɪt̪ɛ", "3", ""),
        ("dʒɔn", "4", ""), ("dʒanha", "2", ""),
    )),
    (43, "sky", (
        ("ket̪ʊŋ", "1", ""), ("ket̪ʊŋ", "1", ""), ("kɪt̪ʊŋ", "1", ""),
        ("akas | bed̪ol", "3|5", "two responses printed on separate lines for KAD"),
        ("bed̪ol", "5", ""),
        ("akas | bed̪ol", "3|5", "two responses printed on separate lines for RAS"),
        ("t̪ɪɾip", "2", ""), ("kɪt̪əŋeni", "1", ""),
        ("t̪eɾu", "4", ""), ("bed̪əl", "5", ""), ("akasu", "3", ""),
    )),
    (44, "star", (
        ("mʊmoɾt̪oŋ", "5", ""), ("mʊmoɾt̪oŋ", "5", ""),
        ("mʊmoɾt̪oŋ", "5", ""), ("momoɾt̪ɔ", "5", ""),
        ("momoɾt̪ɔ", "5", ""),
        ("momoɾt̪ɔ | kimit̪o", "5|6", "two responses printed on separate lines for RAS"),
        ("poʔt̪iŋ", "2", ""), ("tʃaŋkua", "1", ""),
        ("t̪ɛɾɛ", "4", ""), ("t̪ɛɾɛ", "4", ""),
        ("nakʃat̪ɾa | t̪aɾa", "3|4", "two responses printed on separate lines for ODI"),
    )),
    (45, "rain", (
        ("d̪a", "3", ""), ("d̪aʔ", "3", ""), ("d̪aʔ", "3", ""),
        ("d̪ɛguɾoni", "3", ""), ("d̪ɛguɾoni", "3", ""),
        ("d̪ɛguɾoni", "3", ""),
        ("boɾsɛ | d̪ɛ", "2|3", "two responses printed on separate lines for the same two-line GUT label"),
        ("leŋd̪ia", "1", ""), ("d̪ɛ", "3", ""),
        ("pɛnɪ", "4", ""), ("borosa", "2", ""),
    )),
]


def source_pages(item: int, site_code: str) -> tuple[str, str]:
    if item == 41 and site_code in {"POD", "BON", "DUM", "KAD", "KEN"}:
        return "20", "15"
    return "21", "16"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 60
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
