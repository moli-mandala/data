#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 201-205.

Every conceptual cell was independently reviewed from rendered source pages
45-46 at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_201_205_hand_keyed.tsv"
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

# Each cell is (manual transcription, similarity groups, qualification, status).
PAGE_DECISIONS = [
    (201, "look!, he saw", (
        ("dʒu", "2", "", "attested"),
        ("dʒu", "2", "", "attested"),
        ("dʒu", "2", "", "attested"),
        ("dʒu:", "2", "", "attested"),
        ("dʒu:", "2", "", "attested"),
        ("dʒu:", "2", "", "attested"),
        ("dʒu | mɛidʒuvo", "2|2", "two group-2 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("ke", "1", "", "attested"),
        ("gi", "3", "", "attested"),
        ("se d̪eklɛ", "4", "", "attested"),
        ("dekho", "4", "", "attested"),
    )),
    (202, "I (1st sg)", (
        ("niŋ", "1", "", "attested"),
        ("niŋ", "1", "", "attested"),
        ("niŋ", "1", "", "attested"),
        ("niŋ", "1", "", "attested"),
        ("niŋ", "1", "", "attested"),
        ("niŋ", "1", "", "attested"),
        ("niŋ", "1", "", "attested"),
        ("naiŋ", "1", "", "attested"),
        ("mɪŋ", "1", "", "attested"),
        ("mũɪ", "2", "", "attested"),
        ("mũ", "2", "", "attested"),
    )),
    (203, "you (2nd sg, informal)", (
        ("no", "2", "", "attested"),
        ("no", "2", "", "attested"),
        ("no", "2", "", "attested"),
        ("no", "2", "", "attested"),
        ("no", "2", "", "attested"),
        ("no", "2", "", "attested"),
        ("nom", "2", "", "attested"),
        ("na", "1", "", "attested"),
        ("mɛŋ", "3", "", "attested"),
        ("t̪uɪ", "4", "", "attested"),
        ("t̪u", "4", "", "attested"),
    )),
    (204, "you (2nd sg, formal)", (
        ("no", "2", "", "attested"),
        ("no", "2", "", "attested"),
        ("no", "2", "", "attested"),
        ("no", "2", "", "attested"),
        ("", "0", "printed ‘no entry’", "source_blank_no_entry"),
        ("no", "2", "", "attested"),
        ("nom", "2", "", "attested"),
        ("", "0", "printed ‘no entry’", "source_blank_no_entry"),
        ("mɛŋ", "3", "", "attested"),
        ("t̪uɪ", "4", "", "attested"),
        ("ɑponõ", "5", "", "attested"),
    )),
    (205, "he (3rd sg, masculine)", (
        ("mai", "1", "", "attested"),
        ("mai", "1", "", "attested"),
        ("mai", "1", "", "attested"),
        ("mɛi", "1", "", "attested"),
        ("mɛi", "1", "", "attested"),
        ("mɛi", "1", "", "attested"),
        ("mɛi", "1", "", "attested"),
        ("me", "1", "", "attested"),
        ("non", "2", "", "attested"),
        ("se", "3", "", "attested"),
        ("se", "3", "", "attested"),
    )),
]


def source_pages(item: int, site_code: str) -> tuple[str, str]:
    if item == 201 or (item == 202 and site_code in {"POD", "BON", "DUM", "KAD", "KEN"}):
        return "45", "40"
    return "46", "41"


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str, status: str) -> dict[str, str]:
    site_code, site_name, column = site
    pdf_page, printed_page = source_pages(item, site_code)
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": pdf_page,
        "Printed_Page": printed_page, "Column": column,
        "Manual_Transcription": form, "Similarity_Groups": groups,
        "Source_Qualification": qualification, "Review_Status": status,
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
        for site, (form, groups, qualification, status) in zip(SITES, cells, strict=True):
            out.append(make_row(item, gloss, site, form, groups, qualification, status))
    return out


def main() -> None:
    output_rows = rows()
    assert len(output_rows) == 55
    assert len({(row["Item"], row["Site_Code"]) for row in output_rows}) == 55
    assert sum(row["Review_Status"] == "attested" for row in output_rows) == 53
    assert sum(row["Review_Status"] == "source_blank_no_entry" for row in output_rows) == 2
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in output_rows if row["Manual_Transcription"]
    ) == 54
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
