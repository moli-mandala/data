#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 166-170.

Every conceptual cell was independently reviewed from rendered source page 40
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_166_170_hand_keyed.tsv"
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
    (166, "what?", (
        ("ma", "6", ""), ("ma", "6", ""), ("ma", "6", ""),
        ("ma", "6", ""), ("ma", "6", ""), ("ma", "6", ""),
        ("mɛŋt̪e", "2", ""), ("meʔ bare", "1", ""),
        ("nʌɪt̪e", "3", ""), ("kejt̪e", "4", ""),
        ("kɔnɔʔ", "5", ""),
    )),
    (167, "where?", (
        ("aʔri", "7", ""), ("ari", "7", ""), ("ambo", "8", ""),
        ("ʌri", "7", ""), ("ʌri", "7", ""), ("ʌri", "7", ""),
        ("mono", "2", ""), ("and̪i", "1", ""),
        ("t̪ugɛinu", "3", ""), ("kɔnt̪i", "4", ""),
        ("keuntare | kuade", "5|6", "two responses printed on separate group-5 and group-6 lines for ODI"),
    )),
    (168, "when?", (
        ("n̩d̪oi", "2", ""), ("n̩d̪oi", "2", ""),
        ("in̩d̪oja", "2", ""), ("n̩d̪ɔi", "2", ""),
        ("n̩d̪oi", "2", ""), ("n̩d̪ɔi", "2", ""),
        ("ʊn̩d̪oi", "2", ""), ("ʊeʔn̪a", "5", ""),
        ("ʌgɛɪ", "3", ""), ("kebe", "4", ""), ("kebe", "4", ""),
    )),
    (169, "how many?", (
        ("oʔdʒa", "4", ""), ("oidʒa", "4", ""), ("aʔi", "5", ""),
        ("ɔidʒa", "4", ""), ("ɔidʒa", "4", ""), ("ɔidʒa", "4", ""),
        ("ket̪e", "3", ""), ("o:ʔd̪i", "1", ""),
        ("ɪjɛɪ", "2", ""), ("ket̪ət̪e", "3", ""), ("ket̪e", "3", ""),
    )),
    (170, "what kind?", (
        ("miri", "6", ""), ("miri", "6", ""), ("mibai", "6", ""),
        ("mʊrɪ", "6", ""), ("mɪrɪ", "6", ""), ("mɪrɪ", "6", ""),
        ("erɛn", "2", ""), ("d̪eʔd̪irɔkɔm", "1", ""),
        ("ʌmnɛɪʔgɔt̪e", "3", ""), ("ken̪t̪ət̪e", "4", ""),
        ("kemit̪i", "5", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "40", "Printed_Page": "35",
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
