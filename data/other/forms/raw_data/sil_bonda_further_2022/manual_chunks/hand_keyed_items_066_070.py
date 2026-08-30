#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 66-70.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_066_070_hand_keyed.tsv"
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
    (66, "fruit", (
        ("po:lo", "2", "", "attested"),
        ("po:l", "2", "", "attested"),
        ("po:l", "2", "", "attested"),
        ("po:l", "2", "", "attested"),
        ("po:l", "2", "", "attested"),
        ("po:l", "2", "", "attested"),
        ("fol", "2", "", "attested"),
        ("tʃuɖe", "1", "", "attested"),
        ("uɖo", "3", "", "attested"),
        ("phɔl", "2", "", "attested"),
        ("pholo", "2", "", "attested"),
    )),
    (67, "mango", (
        ("ʊli", "1", "", "attested"),
        ("ʊli", "1", "", "attested"),
        ("li", "1", "", "attested"),
        ("ʊli", "1", "", "attested"),
        ("ʊli", "1", "", "attested"),
        ("ʊli", "1", "", "attested"),
        ("bʊlu", "2", "", "attested"),
        ("ʊli", "1", "", "attested"),
        ("uɖo", "3", "", "attested"),
        ("ɛm", "4", "", "attested"),
        ("ɑmbo", "5", "", "attested"),
    )),
    (68, "banana", (
        ("nsʊgʊɖa", "4", "", "attested"),
        ("nsʊʔgʊɖa", "4", "", "attested"),
        ("nsʊʔgʊɖa", "4", "", "attested"),
        ("n̪dʒuɖa", "4", "", "attested"),
        ("n̪dʒuʔnuɖa", "4", "", "attested"),
        ("n̪dʒuɖa", "4", "", "attested"),
        ("khoɖoli", "3", "", "attested"),
        ("n̪so", "1", "", "attested"),
        ("urɛ", "2", "", "attested"),
        ("kɔɖli", "3", "", "attested"),
        ("kodoli", "3", "", "attested"),
    )),
    (69, "wheat", (
        ("gom", "1", "", "attested"),
        ("gom", "1", "", "attested"),
        ("gom", "1", "", "attested"),
        ("gom", "1", "", "attested"),
        ("gom", "1", "", "attested"),
        ("gɔŋ", "1", "", "attested"),
        ("kɛroŋ", "2", "", "attested"),
        ("gɔm", "1", "", "attested"),
        ("", "0", "printed ‘no entry’", "source_blank_no_entry"),
        ("gɔm", "1", "", "attested"),
        ("gohomõ", "3", "", "attested"),
    )),
    (70, "millet", tuple(
        ("", "", "prompt printed DISQUALIFIED", "excluded_disqualified")
        for _ in SITES
    )),
]


def source_pages(item: int, site_code: str) -> tuple[str, str]:
    if item == 66 or (item == 67 and site_code in {"POD", "BON", "DUM", "KAD"}):
        return "24", "19"
    return "25", "20"


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
    assert sum(row["Review_Status"] == "attested" for row in output_rows) == 43
    assert sum(row["Review_Status"] == "source_blank_no_entry" for row in output_rows) == 1
    assert sum(row["Review_Status"] == "excluded_disqualified" for row in output_rows) == 11
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows if row["Manual_Transcription"]) == 43
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
