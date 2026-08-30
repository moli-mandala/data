#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 31-35.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_031_035_hand_keyed.tsv"
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
    (31, "mortar", (
        ("dʒʌn̪t̪a", "1", "source also prints a following group-1 line with no response"),
        ("bire | dʒʌn̪t̪a", "6|1", "source also prints a following group-1 line with no response"),
        ("dʒʌn̪t̪a", "1", "source also prints a following group-1 line with no response"),
        ("sʌŋsʌŋbeɾ", "8", ""), ("sʌŋsʌŋbeɾ", "8", ""),
        ("bɪt̪:morsiŋbire", "7", ""), ("səʔel", "2", ""),
        ("husembaɾ", "1", ""), ("el", "5", ""),
        ("kət̪nɪ", "4", ""),
        ("silɔ | kot̪t̪uni", "3|4", "two responses printed on separate lines for ODI"),
    )),
    (32, "pestle", (
        ("dʒʌn̪t̪a", "1", "source also prints a following group-0 line with no response"),
        ("dʒʌn̪t̪a", "1", "source also prints a following group-0 line with no response"),
        ("dʒʌn̪t̪a", "1", "source also prints a following group-0 line with no response"),
        ("bʌt̪ibeɾ", "7", ""), ("bʌt̪ibeɾ", "7", ""),
        ("d̪aubaibire", "6", ""), ("t̪ɪk:i", "2", ""),
        ("d̪hahusembaɾ", "1", ""), ("indɾɪ", "5", ""),
        ("mʊsəl", "4", ""), ("pothoɾo", "3", ""),
    )),
    (33, "hammer", (
        ("mʊtɭa", "1", ""), ("mʊtɭa", "1", ""), ("mʊtɭa", "1", ""),
        ("mutuɭe", "1", ""), ("mutuɭe", "1", ""), ("mutuɭe", "1", ""),
        ("mʊtɭe", "1", ""), ("mʊtɭa", "1", ""), ("mutɭe", "1", ""),
        ("mʊtɭe", "1", ""), ("hatud̪i", "2", ""),
    )),
    (34, "knife", (
        ("nsʊk", "7", ""), ("nsʊʔ", "7", ""), ("nsʊʔ", "7", ""),
        ("ʊntʃu", "6", ""), ("nsu", "7", ""), ("ʊntʃu", "6", ""),
        ("osʊq", "2", ""), ("suɾisəg˥", "1", ""), ("kʌtɪ", "4", ""),
        ("kɛtɾɛ", "5", ""), ("tʃhuɾi", "3", ""),
    )),
    (35, "axe", (
        ("tʌŋgia", "2", ""), ("tʌŋgia", "2", ""), ("tʌŋgia", "2", ""),
        ("kuɾad̪ɪ", "3", ""), ("kuɾad̪ɪ", "3", ""), ("tʌŋgia", "2", ""),
        ("tɛŋgijɛ", "2", ""), ("malue", "1", ""), ("teŋgɪɛ", "2", ""),
        ("teŋgjɛ", "2", ""), ("taŋgi:a", "2", ""),
    )),
]


def source_pages(item: int, site_code: str) -> tuple[str, str]:
    if item <= 33 or (item == 34 and site_code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS", "GUT"}):
        return "19", "14"
    return "20", "15"


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
