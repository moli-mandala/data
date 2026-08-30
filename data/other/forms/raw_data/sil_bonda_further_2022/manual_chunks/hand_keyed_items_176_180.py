#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 176-180.

Every conceptual cell was independently reviewed from rendered source pages
41-42 at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_176_180_hand_keyed.tsv"
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
    (176, "different", (
        ("pal", "4", "", "attested"),
        ("palpal", "4", "", "attested"),
        ("palpal", "4", "", "attested"),
        ("pal", "4", "", "attested"),
        ("pal", "4", "", "attested"),
        ("pal", "4", "", "attested"),
        ("bɪnbɪn", "1", "", "attested"),
        ("bɪn:ɛ", "1", "", "attested"),
        ("uɾuɾ", "2", "", "attested"),
        ("bɪn", "1", "", "attested"),
        ("bɦino bɦino | ələgɑ", "1|3", "group-1 reduplicated response wraps to the following source line, followed by a group-3 response", "attested"),
    )),
    (177, "whole", (
        ("gʊlai", "5", "", "attested"),
        ("gʊlai", "5", "", "attested"),
        ("gʊlai", "5", "", "attested"),
        ("gʊlai", "5", "", "attested"),
        ("gʊlai", "5", "", "attested"),
        ("gʊlai", "5", "", "attested"),
        ("purʊŋ", "2", "", "attested"),
        ("sʌp:a", "1", "", "attested"),
        ("sɔbu", "3", "", "attested"),
        ("sɔbʊ", "3", "", "attested"),
        ("pura", "4", "", "attested"),
    )),
    (178, "broken", (
        ("waʔguiga", "7", "", "attested"),
        ("waʔguiga", "7", "", "attested"),
        ("waʔguiga", "7", "", "attested"),
        ("puruga", "6", "", "attested"),
        ("puʔruga", "6", "", "attested"),
        ("puruga", "6", "", "attested"),
        ("pɪgu", "2", "", "attested"),
        ("pʌktʃɪ", "1", "", "attested"),
        ("lɪgɛru", "3", "", "attested"),
        ("beŋlɛt̪e", "4", "", "attested"),
        ("boŋgila", "5", "", "attested"),
    )),
    (179, "few", (
        ("uɪt̪ɔjo | una", "1|1", "two comma-separated group-1 responses; source also prints a following group-0 line with no response", "attested"),
        ("uɪt̪ɔjo | una", "1|1", "two comma-separated group-1 responses; source also prints a following group-0 line with no response", "attested"),
        ("iki", "9", "", "attested"),
        ("ɪd̪ikɔŋ", "4", "", "attested"),
        ("ɪd̪ikɔŋ", "4", "", "attested"),
        ("uɪt̪ikɔŋ | una", "7|7", "two comma-separated responses printed on the group-7 line for RAS", "attested"),
        ("kən̪d̪ek", "3", "", "attested"),
        ("ikud̪a | gond̪a", "1|2", "two responses printed on separate group-1 and group-2 lines for BIA", "attested"),
        ("ɪd̪ikon", "4", "", "attested"),
        ("ɔlɔp", "5", "", "attested"),
        ("kom", "6", "", "attested"),
    )),
    (180, "many", (
        ("rept̪e", "6", "", "attested"),
        ("rept̪e", "6", "", "attested"),
        ("rept̪e", "6", "", "attested"),
        ("reʔt̪e | kʊb", "6|8", "two responses printed on separate group-6 and group-8 lines for KAD", "attested"),
        ("reʔt̪e", "6", "", "attested"),
        ("reʔt̪e", "6", "", "attested"),
        ("kət̪:ijo", "2", "", "attested"),
        ("l̪abə", "7", "", "attested"),
        ("ʌbu", "3", "", "attested"),
        ("besɪ", "4", "", "attested"),
        ("bohut", "5", "", "attested"),
    )),
]


def source_page(item: int, site_code: str) -> tuple[str, str]:
    if item == 176 or (item == 177 and site_code in {"POD", "BON", "DUM", "KAD", "KEN", "RAS"}):
        return "41", "36"
    return "42", "37"


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str, status: str) -> dict[str, str]:
    site_code, site_name, column = site
    pdf_page, printed_page = source_page(item, site_code)
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
    assert all(row["Review_Status"] == "attested" for row in output_rows)
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in output_rows if row["Manual_Transcription"]
    ) == 61
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
