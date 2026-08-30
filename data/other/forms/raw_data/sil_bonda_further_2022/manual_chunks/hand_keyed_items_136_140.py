#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 136-140.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_136_140_hand_keyed.tsv"
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
    (136, "hot", (
        ("gige", "2", ""), ("gige", "2", ""), ("gige", "2", ""),
        ("gɛge bai", "2", ""), ("gɛge bai", "2", ""),
        ("gigem", "2", ""),
        ("sileinɖɛ | t̪orlo", "3|4", "responses printed on separate group-3 and group-4 lines for GUT"),
        ("gi", "1", ""), ("bɛlbɛl", "5", ""),
        ("t̪ɔpɔt̪", "6", ""), ("gorɑmo", "7", ""),
    )),
    (137, "cold", (
        ("sep'", "5", ""), ("sep'", "5", ""), ("sep'", "5", ""),
        ("seb", "5", ""), ("seb", "5", ""), ("seb", "5", ""),
        ("ruøo", "2", ""), ("tʃʊi", "1", ""),
        ("kʌkor", "3", ""), ("kɛkɔr", "3", ""),
        ("t̪hanɖa", "4", ""),
    )),
    (138, "right", (
        ("sʊsʊmt̪i", "5", ""), ("sʊsʊmt̪i", "5", ""),
        ("sʊsʊm", "5", ""), ("sʊsʊm", "5", ""),
        ("sʊsʊm", "5", ""), ("sʊsʊm", "5", ""),
        ("et̪om", "2", ""), ("iŋtʃɔŋt̪i", "1", ""),
        ("udʒɛ", "3", ""), ("udʒɛ", "3", ""),
        ("ɖahano", "4", ""),
    )),
    (139, "left", (
        ("bʌsapt̪i", "6", ""), ("basak", "6", ""),
        ("basak", "6", ""),
        ("basɛ | basɛ", "2|6", "same response printed on separate group-2 and group-6 lines for KAD"),
        ("basɛ | basɛ", "2|6", "same response printed on separate group-2 and group-6 lines for KEN"),
        ("basɛ | basɛ", "2|6", "same response printed on separate group-2 and group-6 lines for RAS"),
        ("esɛ", "2", ""), ("beçija", "1", ""),
        ("sinersi", "3", ""), ("ɖebrɪ", "4", ""),
        ("bɑ:mo", "5", ""),
    )),
    (140, "near", (
        ("un̪t̪u", "6", ""), ("n̪t̪u", "6", ""),
        ("n̪t̪u", "6", ""), ("loge", "2", ""),
        ("loge", "2", ""), ("loge", "2", ""),
        ("loge", "2", ""), ("gʊdʒɔ", "4", ""),
        ("loge", "2", ""), ("loge", "2", ""),
        ("pak:o", "3", ""),
    )),
]


def source_page(item: int) -> tuple[str, str]:
    if item <= 137:
        return "35", "30"
    return "36", "31"


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    pdf_page, printed_page = source_page(item)
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
