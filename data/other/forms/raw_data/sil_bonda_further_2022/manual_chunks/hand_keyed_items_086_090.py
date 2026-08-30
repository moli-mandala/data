#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 86-90.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_086_090_hand_keyed.tsv"
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
    (86, "fish", (
        ("aʔɖoŋ", "1", ""), ("aʔɖoŋ", "1", ""),
        ("aʔaɖoŋ", "1", ""), ("aʔaɖoŋ", "1", ""),
        ("aʔaɖoŋ", "1", ""), ("aʔaɖoŋ", "1", ""),
        ("əʔɖoŋ", "1", ""), ("haʔɖe", "2", ""),
        ("ʌju", "3", ""), ("mɛtʃ", "4", ""),
        ("ma:tʃo", "4", ""),
    )),
    (87, "chicken", (
        ("gisiŋ", "1", ""), ("gisiŋ", "1", ""),
        ("gisiŋ", "1", ""), ("giɕiŋ", "1", ""),
        ("giɕiŋ", "1", ""), ("giɕiŋ", "1", ""),
        ("gis:iŋ", "1", ""), ("giseŋ", "1", ""),
        ("ʌŋoi", "2", ""), ("kukrɛ", "3", ""),
        ("kukuda", "3", ""),
    )),
    (88, "egg", (
        ("n̩t̪osiŋ", "1", ""), ("n̩t̪osiŋ", "1", ""),
        ("n̩t̪asiŋ", "1", ""), ("ŋt̪osiŋ", "1", ""),
        ("ŋt̪osiŋ", "1", ""), ("ŋt̪osiŋ", "1", ""),
        ("ʊt̪opsiŋ", "2", ""), ("n̩t̪aseŋ", "1", ""),
        ("ʌɾɪ", "3", ""), ("gɛɾ", "4", ""),
        ("oɳɖa", "5", ""),
    )),
    (89, "cow", (
        ("goɪt̪aŋ | dʒɔŋgoi", "6|6", "two group-6 responses; source qualifier `(female)` applies to the second response"),
        ("goɪt̪aŋ | dʒɔŋgoi", "6|6", "two group-6 responses; source qualifier `(female)` applies to the second response"),
        ("guɪt̪aŋ", "6", ""),
        ("gɔiʔt̪ʌŋ | jɔŋgɔi | gɔiʔt̪ʌŋ", "3|5|6", "source qualifier `(male)` applies to the repeated group-3 and group-6 responses"),
        ("gɔiʔt̪ʌŋ | jɔŋgɔi | gɔiʔt̪ʌŋ", "3|5|6", "source qualifier `(male)` applies to the repeated group-3 and group-6 responses"),
        ("gɔiʔt̪ʌŋ | jɔŋgɔi | gɔiʔt̪ʌŋ", "3|5|6", "source qualifier `(male)` applies to the repeated group-3 and group-6 responses"),
        ("bəŋɖɪ | kɪʈeŋ", "2|3", "two responses printed on separate group-2 and group-3 lines for GUT"),
        ("d̪ijat̪ija", "1", ""), ("kuɪʈeŋ", "3", ""),
        ("gej", "4", ""), ("gai", "4", ""),
    )),
    (90, "buffalo", (
        ("bʊŋʈe | dʒɔŋbʊŋ", "2|2", "two group-2 responses; source qualifier `(female)` applies to the second response"),
        ("bʊŋʈe | dʒɔŋbʊŋ", "2|2", "two group-2 responses; source qualifier `(female)` applies to the second response"),
        ("bʊŋʈe | dʒɔŋbʊŋ", "2|2", "two group-2 responses; source qualifier `(female)` applies to the second response"),
        ("bʊŋʈe | jɔŋbʊŋ", "2|5", "source qualifier `(male)` applies to the group-2 response"),
        ("bʊŋʈe | jɔŋbʊŋ", "2|5", "source qualifier `(male)` applies to the group-2 response"),
        ("bʊŋʈe | jɔŋbʊŋ", "2|5", "source qualifier `(male)` applies to the group-2 response"),
        ("boŋʈel", "2", ""), ("d̪ijabo", "1", ""),
        ("kiboŋ", "3", ""), ("mɔjsɪ", "4", ""),
        ("moi:ʃa", "4", ""),
    )),
]


def source_page(item: int, site_code: str) -> tuple[str, str]:
    if item == 86 or (item == 87 and site_code not in {"PAR", "RON", "ODI"}):
        return "27", "22"
    return "28", "23"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 70
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
