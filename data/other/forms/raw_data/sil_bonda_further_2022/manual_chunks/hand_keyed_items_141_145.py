#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 141-145.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_141_145_hand_keyed.tsv"
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
    (141, "far", (
        ("sʊlʊŋ", "2", ""), ("sʊlʊŋ", "2", ""),
        ("sʊlʊŋ", "2", ""), ("suluŋ", "2", ""),
        ("suluŋ", "2", ""), ("suluŋ", "2", ""),
        ("sʊloŋ", "2", ""), ("slo", "1", ""),
        ("ɖur", "3", ""), ("ɖʊr", "3", ""),
        ("ɖuro", "3", ""),
    )),
    (142, "big", (
        ("mʊna | bʊɖa | mʊna", "1|1|5", "two comma-separated group-1 responses followed by the group-5 response on a separate source line for POD"),
        ("mʊna | bʊɖa | mʊna", "1|1|5", "two comma-separated group-1 responses followed by the group-5 response on a separate source line for BON"),
        ("mʊna | bʊɖa | mʊna", "1|1|5", "two comma-separated group-1 responses followed by the group-5 response on a separate source line for DUM"),
        ("munaʔbai", "5", ""), ("munaʔbai", "5", ""),
        ("munaʔbai", "5", ""), ("moɖo", "2", ""),
        ("mɳa", "1", ""), ("lup", "3", ""),
        ("bɔr", "4", ""), ("boro", "4", ""),
    )),
    (143, "small", (
        ("ɖau", "5", ""), ("ɖau", "5", ""), ("ɖau", "5", ""),
        ("ɖaubai", "5", ""), ("ɖaubai", "5", ""),
        ("ɖaubai", "5", ""), ("miɛn", "2", ""),
        ("ɖãha", "1", ""), ("ʌsu", "3", ""),
        ("sɛn", "4", ""), ("sanõ", "4", ""),
    )),
    (144, "heavy", (
        ("leŋgi", "5", ""), ("lɪŋgi", "1", ""),
        ("leŋgi", "5", ""), ("leŋgɪ", "5", ""),
        ("leŋgɪ", "5", ""), ("bɔdʒ", "2", ""),
        ("lɪgɪŋ | bodʒ", "1|2", "responses printed on separate group-1 and group-2 lines for GUT"),
        ("lɪgɪŋ", "1", ""), ("lʌgun", "3", ""),
        ("bɔdʒ", "2", ""), ("bɦaɾi", "4", ""),
    )),
    (145, "light", (
        ("nijam", "7", ""), ("nijap", "7", ""),
        ("njap", "7", ""), ("olu", "3", ""),
        ("olu", "3", ""), ("ʊsɛs", "2", ""),
        ("ʊsɛs", "2", ""), ("haʒa", "5", ""),
        ("ɔlu", "3", ""), ("usɛs", "2", ""),
        ("haluka", "4", ""),
    )),
]


def source_page(item: int, site_code: str) -> tuple[str, str]:
    if item <= 143 or (item == 144 and site_code in {"POD", "BON", "DUM", "KAD", "KEN"}):
        return "36", "31"
    return "37", "32"


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
