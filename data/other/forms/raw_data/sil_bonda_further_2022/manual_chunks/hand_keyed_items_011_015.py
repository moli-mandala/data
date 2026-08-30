#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 11-15.

Every conceptual cell was independently reviewed from 600-dpi rendered pages
and rechecked in targeted 1200-dpi crops. Item 11 is explicitly disqualified
in the source. OCR, PDF text, and prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_011_015_hand_keyed.tsv"
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

# Each attested cell is (manual transcription, similarity groups, qualification).
PAGE_DECISIONS = [
    (12, "belly", (
        ("sʊloi", "1", ""), ("sʊloi", "1", ""), ("sʊloi", "1", ""),
        ("sʊloi", "1", ""), ("sʊloi", "1", ""), ("sʊloi", "1", ""),
        ("sʊloi", "1", ""), ("sʊle", "1", ""),
        ("pot̪e | put̪e", "3|3", "two responses printed on separate lines for the same two-line PAR label"),
        ("pet̪", "2", ""), ("pet̪t̪o", "4", ""),
    )),
    (13, "arm", (
        ("t̪t̪i", "2", ""), ("t̪t̪i", "2", ""), ("t̪t̪i", "2", ""),
        ("t̪t̪i", "2", ""), ("t̪t̪i", "2", ""), ("t̪t̪i", "2", ""),
        ("t̪t̪i", "2", ""), ("n̩t̪i", "1", ""), ("si", "3", ""),
        ("ɐt̪", "5", ""), ("hat̪o", "4", ""),
    )),
    (14, "elbow", (
        ("sʊnʊkt̪i", "9", ""),
        ("sʊnʊkut̪i | sʊnʊkut̪i", "9|1", "source also prints a following group-0 line with no response"),
        ("sʊnʊkʔt̪i", "9", ""), ("gət̪i", "5", ""),
        ("kəpoɾ", "1", ""), ("gət̪i", "5", ""), ("kop:oɾ", "1", ""),
        ("sekoɾt̪i", "4", ""), ("kumsi", "3", ""), ("kəpəɾ", "1", ""),
        ("koini", "2", ""),
    )),
    (15, "palm", (
        ("git̪at̪i", "7", ""), ("gʊt̪at̪i", "7", ""), ("gʊt̪at̪i", "7", ""),
        ("pəɖomt̪it̪i", "1", ""), ("pəɖomt̪it̪i", "1", ""),
        ("pəɖomt̪i", "1", ""), ("poɖəm", "1", ""),
        ("git̪at̪i", "7", ""), ("si", "5", ""), ("ɐt̪", "6", ""),
        ("toɭohato | papuli", "3|4", "two responses printed on separate lines"),
    )),
]


def source_pages(item: int, site_code: str) -> tuple[str, str]:
    if item <= 12 or (item == 13 and site_code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS"}):
        return "16", "11"
    return "17", "12"


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
        "Reviewed_At": "2026-08-28", "Reviewer_Declaration": DECLARATION,
    }
    assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
    return row


def rows() -> list[dict[str, str]]:
    out = [
        make_row(
            11, "breast", site, "", "",
            "prompt printed DISQUALIFIED across all eleven lists",
            "excluded_disqualified",
        )
        for site in SITES
    ]
    for item, gloss, cells in PAGE_DECISIONS:
        assert len(cells) == len(SITES) == 11
        for site, (form, groups, qualification) in zip(SITES, cells, strict=True):
            out.append(make_row(item, gloss, site, form, groups, qualification, "attested"))
    return out


def main() -> None:
    output_rows = rows()
    assert len(output_rows) == 55
    assert len({(row["Item"], row["Site_Code"]) for row in output_rows}) == 55
    assert sum(row["Review_Status"] == "excluded_disqualified" for row in output_rows) == 11
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows if row["Manual_Transcription"]) == 47
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
