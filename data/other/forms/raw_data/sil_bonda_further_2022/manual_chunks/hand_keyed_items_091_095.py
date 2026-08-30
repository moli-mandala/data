#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 91-95.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_091_095_hand_keyed.tsv"
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
    (91, "milk", (
        ("d̪at̪ʊkʊi", "6", ""), ("d̪at̪ʊkʊi", "6", ""),
        ("d̪at̪ikʊi", "6", ""), ("d̪at̪ukui", "6", ""),
        ("d̪at̪ukui", "6", ""), ("d̪at̪ukui", "6", ""),
        ("d̪əkt̪or", "2", ""), ("t̪ra", "1", ""),
        ("ʌɖu", "3", ""), ("ɖʊɖ", "4", ""),
        ("khiro", "5", ""),
    )),
    (92, "horns", (
        ("ɖɔrʊŋ", "1", ""), ("ɖerʊŋ", "1", ""),
        ("ɖerʊŋ", "1", ""), ("ɖerʊŋ", "1", ""),
        ("ɕiŋ", "3", ""), ("ɕiŋ", "3", ""),
        ("siŋ", "3", ""), ("ɖiraŋ", "1", ""),
        ("ɖʌru", "2", ""), ("siŋ", "3", ""),
        ("siŋgə", "3", ""),
    )),
    (93, "tail", (
        ("pʊra", "2", ""), ("pʊra", "2", ""),
        ("pʊra", "2", ""), ("pʊrɛ", "2", ""),
        ("pʊrɛ", "2", ""), ("pʊrɛ", "2", ""),
        ("pʊɖɛ", "2", ""), ("pɭa", "1", ""),
        ("leŋdʒ", "3", ""), ("leŋdʒ", "3", ""),
        ("landʒo", "3", ""),
    )),
    (94, "goat", (
        ("gime", "1", ""), ("gime", "1", ""),
        ("gime", "1", ""), ("gime", "1", ""),
        ("gime", "1", ""), ("gime", "1", ""),
        ("gim:ɛ", "1", ""), ("gimi", "1", ""),
        ("kinme", "2", ""), ("tʃeli", "3", ""),
        ("tʃheli", "3", ""),
    )),
    (95, "dog", (
        ("gʊso", "1", ""), ("gʊso", "1", ""),
        ("gʊsʊ", "1", ""), ("guso", "1", ""),
        ("guso", "1", ""), ("guso", "1", ""),
        ("gʊso", "1", ""), ("gʊsu", "1", ""),
        ("kusor", "2", ""), ("kʊkʊr", "3", ""),
        ("kukurɑ", "3", ""),
    )),
]


def source_page(item: int) -> tuple[str, str]:
    return ("28", "23") if item <= 92 else ("29", "24")


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 55
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
