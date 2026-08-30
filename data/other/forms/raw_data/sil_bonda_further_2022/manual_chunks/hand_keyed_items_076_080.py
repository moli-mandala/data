#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 76-80.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_076_080_hand_keyed.tsv"
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
    (76, "turmeric", (
        ("sʌŋsʌŋ", "2", ""), ("sʌŋsʌŋ", "2", ""),
        ("sʌŋsʌŋ", "2", ""), ("sʌŋsʌŋ", "2", ""),
        ("sʌŋsʌŋ", "2", ""), ("sʌŋsʌŋ", "2", ""),
        ("sɛŋsɛŋ", "2", ""), ("çiçia", "1", ""),
        ("sʌŋsɛŋ", "2", ""), ("ɔlɖɪ", "3", ""),
        ("holidi", "3", ""),
    )),
    (77, "garlic", (
        ("t̪ʊlirʊsuɳo", "3", ""), ("t̪ʊlirʊsuɳo", "3", ""),
        ("t̪ʊlirʊsuɳo", "3", ""), ("rusʊɳo", "2", ""),
        ("rusʊɳo", "2", ""), ("rusʊɳ", "2", ""),
        ("loson", "2", ""), ("t̪ulaisulei", "1", ""),
        ("losun", "2", ""), ("lɔsʊn", "2", ""),
        ("rəsuɳə", "2", ""),
    )),
    (78, "onion", (
        ("ʊlirusuno", "3", ""), ("ʊlirusuno", "3", ""),
        ("ʊlirusuno", "3", ""), ("piɛdʒ", "2", ""),
        ("piɛdʒ", "2", ""), ("piɛdʒ", "2", ""),
        ("pijɛdʒ | ʊl:i", "2|3", "two responses printed on separate group-2 and group-3 lines for GUT"),
        ("tʃisulei", "1", ""), ("piɛdʒ", "2", ""),
        ("uli", "3", ""), ("piadʒo", "2", ""),
    )),
    (79, "cauliflower", (
        ("pʊlkobi", "2", ""), ("pʊlkobi", "2", ""),
        ("kubi", "2", ""), ("kubi", "3", ""),
        ("kubi", "3", ""), ("kubi", "3", ""),
        ("phulkobi", "2", ""), ("sarikobi", "1", ""),
        ("pulkobi", "2", ""), ("phulkɔbi", "2", ""),
        ("phulkobi", "2", ""),
    )),
    (80, "tomato", (
        ("bedʒa", "2", ""), ("bedʒa", "2", ""),
        ("bedʒa", "2", ""), ("bedʒɪrɪ", "1", ""),
        ("bedʒɪrɪ", "1", ""), ("bedʒɪrɪ", "1", ""),
        ("bedʒɪri", "1", ""), ("bedʒa", "2", ""),
        ("bedʒrɪ", "1", ""), ("bedʒrɪ", "1", ""),
        ("bilaʈi", "3", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "26", "Printed_Page": "21",
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
