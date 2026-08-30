#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 206-210.

Every conceptual cell was independently reviewed from rendered source pages
46-47 at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_206_210_hand_keyed.tsv"
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
    (206, "she (3rd sg, feminine)", (
        ("mai", "2", "", "attested"),
        ("mai", "2", "", "attested"),
        ("mai", "2", "", "attested"),
        ("mɛi", "2", "", "attested"),
        ("mɛi", "2", "", "attested"),
        ("mɛi", "2", "", "attested"),
        ("mɛi", "2", "", "attested"),
        ("merɑ", "1", "", "attested"),
        ("non", "3", "", "attested"),
        ("se", "4", "", "attested"),
        ("se", "4", "", "attested"),
    )),
    (207, "we (1st pl, inclusive)", (
        ("nai", "1", "", "attested"),
        ("nai", "1", "", "attested"),
        ("nai", "1", "", "attested"),
        ("ne", "1", "", "attested"),
        ("ne", "1", "", "attested"),
        ("ne", "1", "", "attested"),
        ("nɛinen", "2", "", "attested"),
        ("ne", "1", "", "attested"),
        ("bilɛŋ", "3", "", "attested"),
        ("ɛmɛmɔn", "4", "", "attested"),
        ("ɑme | ɑmpe", "5|5", "two group-5 responses printed on consecutive lines", "attested"),
    )),
    (208, "we (1st pl, exclusive)", (
        ("nai", "2", "", "attested"),
        ("nai", "2", "", "attested"),
        ("nai", "2", "", "attested"),
        ("ne", "2", "", "attested"),
        ("ne", "2", "", "attested"),
        ("naŋ", "4", "", "attested"),
        ("nei", "2", "", "attested"),
        ("ok:en remo", "1", "", "attested"),
        ("", "0", "printed ‘no entry’", "source_blank_no_entry"),
        ("ɛme", "3", "", "attested"),
        ("ɑme | ɑmpe", "3|3", "two group-3 responses printed on consecutive lines", "attested"),
    )),
    (209, "you (2nd pl)", (
        ("pe", "1", "", "attested"),
        ("pe", "1", "", "attested"),
        ("pe", "1", "", "attested"),
        ("pele", "1", "", "attested"),
        ("pe", "1", "", "attested"),
        ("pe", "1", "", "attested"),
        ("pɛn", "1", "", "attested"),
        ("pe", "1", "", "attested"),
        ("mɛŋdʒɪ", "2", "", "attested"),
        ("t̪ume", "3", "", "attested"),
        ("ɑponõ", "4", "", "attested"),
    )),
    (210, "they (3rd pl)", (
        ("meʔje", "2", "", "attested"),
        ("maʔje", "2", "", "attested"),
        ("meʔje", "2", "", "attested"),
        ("maʔɪle", "2", "", "attested"),
        ("maʔɛ:", "2", "", "attested"),
        ("maʔɪle", "2", "", "attested"),
        ("mɛinen", "3", "", "attested"),
        ("mehɲ", "1", "", "attested"),
        ("ʌd̪iŋmɔɪ", "4", "", "attested"),
        ("semɔn", "5", "", "attested"),
        ("se mɑn:e", "6", "", "attested"),
    )),
]


def source_pages(item: int) -> tuple[str, str]:
    return ("46", "41") if item <= 208 else ("47", "42")


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str, status: str) -> dict[str, str]:
    site_code, site_name, column = site
    pdf_page, printed_page = source_pages(item)
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
    assert sum(row["Review_Status"] == "attested" for row in output_rows) == 54
    assert sum(row["Review_Status"] == "source_blank_no_entry" for row in output_rows) == 1
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in output_rows if row["Manual_Transcription"]
    ) == 56
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
