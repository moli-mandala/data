#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for physical p. 16, items 6-10.

Every response and similarity-group label below was independently hand-keyed
from the rendered source at 600 dpi and rechecked in targeted 1200-dpi crops.
PDF extraction, OCR, and the earlier report are not transcription inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_006_010_hand_keyed.tsv"
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

# Each cell is (manual transcription, printed similarity group, qualification).
PAGE_DECISIONS = [
    (6, "ear", (
        ("lʊn̩t̪ʊr", "5", ""), ("lʊn̩t̪ʊr", "5", ""), ("lʊn̩t̪ʊr", "5", ""),
        ("lunt̪ur", "5", ""), ("lunt̪ur", "5", ""), ("lunt̪ur", "5", ""),
        ("l̩t̪iɾ", "3", ""),
        ("n̩lʊg̚", "2", "source prints combining left angle above on final g (unreleased)"),
        ("lũ", "4", ""), ("kan", "1", ""), ("kaɳo", "1", ""),
    )),
    (7, "nose", (
        ("nsemi", "4", ""), ("nsemi", "4", ""), ("nsemi", "4", ""),
        ("n̩tʃemi", "4", ""), ("n̩tʃeʔmui", "4", ""), ("n̩tʃemi", "4", ""),
        ("mi", "3", ""), ("mu", "1", ""), ("mũ", "1", ""),
        ("nak", "2", ""), ("nakho", "2", ""),
    )),
    (8, "mouth", (
        ("t̪ʊmo", "1", ""), ("t̪ʊmo", "1", ""), ("t̪ʊmo", "1", ""),
        ("t̪ʊmo", "1", ""), ("t̪umo", "1", ""), ("t̪ʊmo", "1", ""),
        ("t̪ʊmo", "1", ""), ("t̪ʊmua", "1", ""), ("t̪oɾ", "3", ""),
        ("ʈɔɳɖ", "4", ""), ("paʈ:i", "2", ""),
    )),
    (9, "tooth", (
        ("gine", "1", ""), ("gine", "1", ""), ("gine", "1", ""),
        ("gine", "1", ""), ("gine", "1", ""), ("gine", "1", ""),
        ("ginɛ", "1", ""), ("gini", "1", ""), ("dʒi", "3", ""),
        ("d̪ẽt̪", "2", ""), ("d̪ant̪o", "2", ""),
    )),
    (10, "tongue", (
        ("lejʌŋ", "4", ""), ("leʔjʌŋ", "4", ""), ("leʔjʌŋ", "4", ""),
        ("lejʌŋ", "4", ""), ("lejʌŋ", "4", ""), ("lejʌŋ", "4", ""),
        ("lɐø", "1", ""), ("n̩lia", "3", ""), ("lɐŋ", "1", ""),
        ("dʒɪb", "2", ""), ("dʒibhə", "2", ""),
    )),
]


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item, gloss, cells in PAGE_DECISIONS:
        assert len(cells) == len(SITES) == 11
        for (site_code, site_name, column), (form, group, qualification) in zip(SITES, cells, strict=True):
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
                "Site_Name": site_name, "PDF_Page": "16", "Printed_Page": "11",
                "Column": column, "Manual_Transcription": form,
                "Similarity_Groups": group, "Source_Qualification": qualification,
                "Review_Status": "attested", "Confidence": "high", "Uncertainty": "",
                "Reviewer_Method": "manual visual inspection at 600 dpi; every cell rechecked in targeted 1200-dpi crops",
                "Reviewed_At": "2026-08-28", "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            out.append(row)
    return out


def main() -> None:
    output_rows = rows()
    assert len(output_rows) == 55
    assert len({(row["Item"], row["Site_Code"]) for row in output_rows}) == 55
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-keyed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
