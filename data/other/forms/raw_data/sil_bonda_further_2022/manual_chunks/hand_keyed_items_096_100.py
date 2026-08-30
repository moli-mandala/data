#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 96-100.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_096_100_hand_keyed.tsv"
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
    (96, "snake", (
        ("bʊbʊ", "2", ""), ("bʊbʊ", "2", ""),
        ("bʊbʊ", "2", ""), ("bʊbʊ", "2", ""),
        ("bʊbʊ", "2", ""), ("bʊbu", "2", ""),
        ("bʊɖboi", "1", ""), ("bubo", "2", ""),
        ("bubur", "2", ""), ("sɛp", "3", ""),
        ("sapo", "3", ""),
    )),
    (97, "monkey", (
        ("gisa", "4", ""), ("gisaʔ", "4", ""),
        ("gisa", "4", ""), ("gisa", "4", ""),
        ("gisa", "4", ""), ("gisa", "4", ""),
        ("məkoɖ", "2", ""), ("giɕija", "1", ""),
        ("mʌkuɖɪ", "2", ""), ("mɛkɔr", "2", ""),
        ("maŋkərə", "3", ""),
    )),
    (98, "mosquito", (
        ("kirinje", "5", ""), ("kirinje", "5", ""),
        ("kirinje", "5", ""), ("kirinjẽ", "5", ""),
        ("kirinjẽ", "5", ""), ("kirinjẽ", "5", ""),
        ("bʊrsʊnɖi", "2", ""), ("kʊrni", "1", ""),
        ("bursu", "2", ""), ("tʃabra", "3", ""),
        ("motʃha", "4", ""),
    )),
    (99, "ant", (
        ("bʊje", "5", ""), ("bʊje", "5", ""),
        ("bʊje", "5", ""), ("bije", "5", ""),
        ("bije", "5", ""), ("bije", "5", ""),
        ("gʊnelo", "1", ""),
        ("giɳaluo | buhi", "1|6", "two responses printed on separate group-1 and group-6 lines for BIA"),
        ("mumui", "2", ""), ("tʃɛtɪ", "3", ""),
        ("matʃi", "4", ""),
    )),
    (100, "spider", (
        ("siranguli", "6", ""), ("sirangulugoi", "6", ""),
        ("serangulugoi", "6", ""), ("makaɖa", "5", ""),
        ("makaɖa", "5", ""), ("makaɖa", "5", ""),
        ("kokoŋɖɛ | pɛt̪məkʊɖɪ", "2|3", "two responses printed on separate group-2 and group-3 lines for GUT"),
        ("bʌli", "1", ""), ("pɛt̪mɛkɖɪ", "3", ""),
        ("pɛt̪mɛkrɛ", "3", ""), ("buɖhiɑɳɪ", "4", ""),
    )),
]


def source_page(item: int, site_code: str) -> tuple[str, str]:
    if item <= 98 or (item == 99 and site_code not in {"PAR", "RON", "ODI"}):
        return "29", "24"
    return "30", "25"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 57
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
