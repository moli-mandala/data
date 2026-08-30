#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 171-175.

Every conceptual cell was independently reviewed from rendered source page 41
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_171_175_hand_keyed.tsv"
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
    (171, "this", (
        ("koʔn̪a", "5", "", "attested"),
        ("koʔn̪a", "5", "", "attested"),
        ("kena", "5", "", "attested"),
        ("kɔna", "5", "", "attested"),
        ("kɔʔn̪a", "5", "", "attested"),
        ("kɔna", "5", "", "attested"),
        ("enu", "2", "", "attested"),
        ("kenʌe", "1", "", "attested"),
        ("ɪd̪ɪn", "3", "", "attested"),
        ("et̪e", "4", "", "attested"),
        ("eit̪a", "4", "", "attested"),
    )),
    (172, "that", (
        ("gʊron̪a", "6", "", "attested"),
        ("gʊtʊna | gʊtʊna", "1|6", "identical responses printed on separate group-1 and group-6 lines for BON", "attested"),
        ("gʊtʊna | gʊtʊna", "1|6", "identical responses printed on separate group-1 and group-6 lines for DUM", "attested"),
        ("gɪna", "5", "", "attested"),
        ("gɪt̪ina", "1", "", "attested"),
        ("gɪna", "5", "", "attested"),
        ("onu", "2", "", "attested"),
        ("gɪt̪ene", "1", "", "attested"),
        ("ʌd̪ɪn", "3", "", "attested"),
        ("sɪt̪e", "4", "", "attested"),
        ("seit̪a", "4", "", "attested"),
    )),
    (173, "these", (
        ("koʔn̪a gʊlai", "8", "", "attested"),
        ("koʔn̪a gʊlai", "8", "", "attested"),
        ("", "0", "printed ‘no entry’", "source_blank_no_entry"),
        ("kɔn̪le", "6", "", "attested"),
        ("kɔn̪le", "6", "", "attested"),
        ("kɔn̪le", "6", "", "attested"),
        ("enu", "2", "", "attested"),
        ("khen̪iŋ", "1", "", "attested"),
        ("ɪd̪ɪn", "3", "", "attested"),
        ("ɛt̪emɔn", "4", "", "attested"),
        ("eisʌbu", "5", "", "attested"),
    )),
    (174, "those", (
        ("gʊron̪a gʊlai", "7", "", "attested"),
        ("gʊron̪a gʊlai", "7", "", "attested"),
        ("", "0", "printed ‘no entry’", "source_blank_no_entry"),
        ("gɪnle", "5", "", "attested"),
        ("gɪt̪enle", "6", "", "attested"),
        ("gɪn̪le", "5", "", "attested"),
        ("ʊn:u", "1", "", "attested"),
        ("gɪt̪eniŋ", "6", "", "attested"),
        ("ʌd̪ɪn", "2", "", "attested"),
        ("sɪt̪emɔn", "3", "", "attested"),
        ("seisʌbu", "4", "", "attested"),
    )),
    (175, "same", (
        ("saman", "1", "", "attested"),
        ("sʌman", "1", "", "attested"),
        ("saman", "1", "", "attested"),
        ("saman", "1", "", "attested"),
        ("saman", "1", "", "attested"),
        ("saman", "1", "", "attested"),
        ("səm:ɛn | somɛn", "1|1", "two group-1 responses printed on consecutive lines for GUT", "attested"),
        ("saman", "1", "", "attested"),
        ("ekepere", "2", "", "attested"),
        ("sɔmɛn", "1", "", "attested"),
        ("səman", "1", "", "attested"),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str, status: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "41", "Printed_Page": "36",
        "Column": column, "Manual_Transcription": form,
        "Similarity_Groups": groups, "Source_Qualification": qualification,
        "Review_Status": status, "Confidence": "high", "Uncertainty": "",
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
    assert sum(row["Review_Status"] == "attested" for row in output_rows) == 53
    assert sum(row["Review_Status"] == "source_blank_no_entry" for row in output_rows) == 2
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
