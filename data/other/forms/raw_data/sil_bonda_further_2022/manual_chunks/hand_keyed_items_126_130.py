#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 126-130.

Every conceptual cell was independently reviewed from the rendered source
page at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_126_130_hand_keyed.tsv"
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
    (126, "month", (
        ("ʌrke", "1", ""), ("ʌrke", "1", ""),
        ("arke | masek | mesek", "1|1|3", "two comma-separated responses printed on the group-1 line plus one response on the group-3 line for DUM"),
        ("mesek", "3", ""), ("mesek", "3", ""),
        ("mesek", "3", ""),
        ("mes | mes", "2|3", "same response printed on separate group-2 and group-3 lines for GUT"),
        ("arke", "1", ""),
        ("mes | mes", "2|3", "same response printed on separate group-2 and group-3 lines for PAR"),
        ("mes | mes", "2|3", "same response printed on separate group-2 and group-3 lines for RON"),
        ("masoʔ", "2", ""),
    )),
    (127, "year", (
        ("bɔrsek", "2", ""), ("bɔrsek", "2", ""),
        ("bɔrsek", "2", ""), ("bɔrɛs", "2", ""),
        ("bɔrsek", "2", ""), ("bɔrsek", "2", ""),
        ("borsek", "2", ""), ("mimʊa", "1", ""),
        ("bɔrs", "2", ""), ("bɔrɔs", "2", ""),
        ("bərsə", "2", ""),
    )),
    (128, "old", (
        ("bʌjur", "2", ""), ("bʌjur", "2", ""),
        ("baiur", "2", ""), ("bair", "2", ""),
        ("bair", "2", ""), ("bairga", "2", ""),
        ("bəbɪr", "3", ""), ("bapɪrne", "1", ""),
        ("pɛpur", "4", ""), ("pʊrnɛ", "5", ""),
        ("poruɳɑ", "5", ""),
    )),
    (129, "new", (
        ("t̪ime", "1", ""), ("t̪ime", "1", ""),
        ("t̪ime", "1", ""), ("t̪ime", "1", ""),
        ("t̪ime", "1", ""), ("t̪ime beʔna", "1", ""),
        ("t̪imɛ", "1", ""), ("t̪imine", "1", ""),
        ("t̪emi", "1", ""), ("nũɛ̃", "2", ""),
        ("nu:ɑ̃", "2", ""),
    )),
    (130, "good", (
        ("bol", "5", ""), ("bol", "5", ""),
        ("bol", "5", ""), ("nimɛn", "2", ""),
        ("nimɛn", "2", ""), ("nimɛn", "2", ""),
        ("nimɛn", "2", ""),
        ("imanɖa | bɔl", "1|5", "responses printed on separate group-1 and group-5 lines for BIA"),
        ("nimɛn", "2", ""), ("nikɔ", "3", ""),
        ("bɦolo", "4", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "34", "Printed_Page": "29",
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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 61
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
