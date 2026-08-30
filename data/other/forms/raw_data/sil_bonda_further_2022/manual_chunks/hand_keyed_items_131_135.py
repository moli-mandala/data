#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 131-135.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_131_135_hand_keyed.tsv"
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
    (131, "bad", (
        ("bolʌra | olianɖra | bolʌra", "6|8|8", "group-6 response plus two comma-separated group-8 responses, with the second wrapped onto the following source line for POD"),
        ("bolʌra | olianɖra | bolʌra", "6|8|8", "group-6 response plus two comma-separated group-8 responses, with the second wrapped onto the following source line for BON"),
        ("olira", "8", ""), ("bolanɖa", "6", ""),
        ("bolanɖa", "6", ""), ("nimananɖa", "5", ""),
        ("bənije", "3", ""), ("ɖoɖija", "1", ""),
        ("nʌse", "2", ""), ("bənije", "3", ""),
        ("kɑrɑpo", "4", ""),
    )),
    (132, "wet", (
        ("daʔgɖa", "6", ""), ("daʔgɖa", "6", ""),
        ("ɖagra", "6", ""), ("ɖagra", "6", ""),
        ("ɖagra", "6", ""), ("ɖagra", "6", ""),
        ("bʊgɖɛ", "2", ""),
        ("brɔnɖe | lobonle", "1|7", "responses printed on separate group-1 and group-7 lines for BIA"),
        ("ɖiɖɛ", "3", ""), ("bidʒlɛtɛ", "4", ""),
        ("oɖa", "5", ""),
    )),
    (133, "dry", (
        ("nsor", "2", ""), ("nsor", "2", ""),
        ("nsor", "2", ""), ("n̩dʒor", "5", ""),
        ("n̩dʒor", "5", ""), ("n̩dʒor", "5", ""),
        ("ʊsor", "2", ""), ("n̩sʊar", "1", ""),
        ("ʌsɛr", "2", ""), ("sʊklɛtɛ", "3", ""),
        ("sukhila", "4", ""),
    )),
    (134, "long", (
        ("silai", "1", ""), ("silai", "1", ""),
        ("sileim", "1", ""), ("çileŋ", "1", ""),
        ("çileŋ", "1", ""), ("çileŋ", "1", ""),
        ("silei", "1", ""), ("tʃilɛ", "1", ""),
        ("ɖuŋkɛ", "2", ""), ("ɖeŋ", "3", ""),
        ("lomba", "4", ""),
    )),
    (135, "short", (
        ("ɖilei", "1", ""), ("ɖilei", "1", ""),
        ("ɖilei", "1", ""),
        ("ɖileboi | tʃorko baina", "1|4", "responses printed on separate group-1 and group-4 lines for KAD"),
        ("ɖileboi | tʃorko bai", "1|4", "responses printed on separate group-1 and group-4 lines for KEN"),
        ("tʃorko baina", "4", ""), ("ɖilei", "1", ""),
        ("ɖle", "1", ""), ("ɖelɪ", "1", ""),
        ("tʃɔt", "2", ""), ("tsot̪ia", "3", ""),
    )),
]


def source_page(item: int, site_code: str) -> tuple[str, str]:
    if item == 131 and site_code not in {"RON", "ODI"}:
        return "34", "29"
    return "35", "30"


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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 62
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
