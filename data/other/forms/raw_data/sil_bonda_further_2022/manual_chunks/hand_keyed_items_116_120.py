#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 116-120.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_116_120_hand_keyed.tsv"
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
    (116, "girl", (
        ("selane", "7", "", "attested"), ("selane", "7", "", "attested"),
        ("selane", "7", "", "attested"), ("ɖakui", "6", "", "attested"),
        ("ɖakui", "6", "", "attested"), ("ɖakui", "6", "", "attested"),
        ("onoʔon", "2", "", "attested"), ("selamboʔo", "1", "", "attested"),
        ("guɳɪ", "3", "", "attested"), ("tɔkɪ", "4", "", "attested"),
        ("dʒiopilɑʔ", "5", "", "attested"),
    )),
    (117, "day", (
        ("simi", "1", "", "attested"), ("simi", "1", "", "attested"),
        ("simiʔɪ", "1", "", "attested"), ("çimiʔi", "1", "", "attested"),
        ("çimiʔi", "1", "", "attested"), ("çimiʔi", "1", "", "attested"),
        ("simin", "1", "", "attested"), ("çimi", "1", "", "attested"),
        ("dʒɛɖɪŋ", "2", "", "attested"), ("ɖɪn", "3", "", "attested"),
        ("ɖino", "3", "", "attested"),
    )),
    (118, "night", (
        ("t̪umungo", "4", "", "attested"), ("t̪umungo", "4", "", "attested"),
        ("t̪umungo", "4", "", "attested"), ("minɖip", "1", "", "attested"),
        ("minɖip", "1", "", "attested"), ("minɖip", "1", "", "attested"),
        ("noiʔjel", "2", "", "attested"), ("miɖig", "1", "", "attested"),
        ("", "0", "printed ‘no entry’", "source_blank_no_entry"),
        ("rɛʈɪ", "3", "", "attested"), ("rɑʈi", "3", "", "attested"),
    )),
    (119, "morning", (
        ("ndʒur", "1", "", "attested"), ("ndʒur", "1", "", "attested"),
        ("ndʒur", "1", "", "attested"), ("n̩dʒur", "1", "", "attested"),
        ("n̩dʒur", "1", "", "attested"), ("n̩dʒur", "1", "", "attested"),
        ("oŋdʒirel", "2", "", "attested"), ("n̩dʒɪr", "1", "", "attested"),
        ("sʌkel", "3", "", "attested"), ("sɛkel", "3", "", "attested"),
        ("səkalə", "3", "", "attested"),
    )),
    (120, "noon", (
        ("simi", "1", "", "attested"), ("simi", "1", "", "attested"),
        ("simiʔɪ", "1", "", "attested"), ("mʊnɖebel", "5", "", "attested"),
        ("mʊnɖebel", "5", "", "attested"), ("mʊnɖebel", "5", "", "attested"),
        ("simin | ɛɖʊbelɛ", "1|2", "two responses printed on separate group-1 and group-2 lines for GUT", "attested"),
        ("çimi", "1", "", "attested"), ("dʒɛɖɪŋʔɛɪ", "3", "", "attested"),
        ("ɛrbelɛ", "2", "", "attested"), ("məd̪hjan:ə", "4", "", "attested"),
    )),
]


def source_page(item: int) -> tuple[str, str]:
    return ("32", "27") if item <= 118 else ("33", "28")


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str, status: str) -> dict[str, str]:
    site_code, site_name, column = site
    pdf_page, printed_page = source_page(item)
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
    assert sum(row["Review_Status"] == "attested" for row in output_rows) == 54
    assert sum(row["Review_Status"] == "source_blank_no_entry" for row in output_rows) == 1
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows if row["Manual_Transcription"]) == 55
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
