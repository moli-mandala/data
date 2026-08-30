#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 151-155.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_151_155_hand_keyed.tsv"
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
    (151, "one", (
        ("mʊjõ", "1", ""), ("mʊjõ", "1", ""),
        ("mʊjʊ", "1", ""), ("mujɔŋ", "1", ""),
        ("mujɔŋ", "1", ""), ("mujɔŋ", "1", ""),
        ("mɪro", "2", ""), ("muiŋ", "1", ""),
        ("boɪ", "3", ""), ("gətek", "4", ""),
        ("eko", "5", ""),
    )),
    (152, "two", (
        ("mbar", "1", ""), ("mbar", "1", ""),
        ("mbaru", "1", ""), ("mbaʔar", "1", ""),
        ("mbaʔar", "1", ""), ("mbaʔar", "1", ""),
        ("dʒoɖe", "2", ""), ("mbar", "1", ""),
        ("bɛg", "3", ""), ("dʒɔek", "4", ""),
        ("du:i", "5", ""),
    )),
    (153, "three", (
        ("ŋgɪ", "4", ""), ("ŋgɪ", "4", ""),
        ("iʔŋge", "4", ""), ("t̪in̪t̪a", "3", ""),
        ("t̪in̪t̪a", "3", ""), ("t̪in̪t̪a", "3", ""),
        ("t̪in", "3", ""), ("n̪dʒi", "1", ""),
        ("jɛg", "2", ""), ("t̪in", "3", ""),
        ("t̪ini", "3", ""),
    )),
    (154, "four", (
        ("ũʔũ", "5", ""), ("ũʔũ", "5", ""),
        ("ũʔũ", "5", ""), ("tʃɛrta", "2", ""),
        ("tʃɛrta", "2", ""), ("tʃɛrta", "2", ""),
        ("tʃɛri", "2", ""), ("õ", "1", ""),
        ("uŋgɪ", "3", ""), ("tʃɛr", "2", ""),
        ("tʃaɾi | tʃaɾgo", "2|2", "responses printed on separate group-2 lines for ODI"),
    )),
    (155, "five", (
        ("past̪a", "2", ""), ("past̪a", "2", ""),
        ("moloi", "3", ""), ("pɛndʒt̪a", "2", ""),
        ("pɛndʒt̪a", "2", ""), ("pɛndʒt̪a", "2", ""),
        ("pɛndʒ", "2", ""), ("male", "1", ""),
        ("pɛntʃ", "2", ""), ("pɛntʃ", "2", ""),
        ("pantʃə", "2", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "38", "Printed_Page": "33",
        "Column": column, "Manual_Transcription": form,
        "Similarity_Groups": groups, "Source_Qualification": qualification,
        "Review_Status": "attested", "Confidence": "high", "Uncertainty": "",
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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 56
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
