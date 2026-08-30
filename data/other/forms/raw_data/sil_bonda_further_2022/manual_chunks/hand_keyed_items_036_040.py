#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 36-40.

Every conceptual cell was independently reviewed from the rendered source at
600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_036_040_hand_keyed.tsv"
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

# Each cell is (manual transcription, similarity group, qualification).
PAGE_DECISIONS = [
    (36, "rope", (
        ("gige", "1", ""), ("gige", "1", ""), ("gigei", "1", ""),
        ("gie", "1", ""), ("gie", "1", ""), ("gie", "1", ""),
        ("gɛi", "1", ""), ("gɛhe", "1", ""), ("luʊeɾ", "4", ""),
        ("d̪əɾ", "2", ""), ("dowudi", "3", ""),
    )),
    (37, "thread", (
        ("sut̪a", "1", ""), ("sut̪a", "1", ""), ("sut̪a", "1", ""),
        ("sut̪a", "1", ""), ("sut̪a", "1", ""), ("sut̪a", "1", ""),
        ("sot̪ɛ", "1", ""), ("sut̪a", "1", ""), ("sut̪ɛ", "1", ""),
        ("sot̪ɛ", "1", ""), ("su:t̪a", "1", ""),
    )),
    (38, "needle", (
        ("sudʒa", "1", ""), ("sʊdʒɪ", "1", ""), ("sʊdʒɪ", "1", ""),
        ("sudʒɛ", "1", ""), ("sudʒɪ", "1", ""), ("sudʒɪ", "1", ""),
        ("sʊdʒi", "1", ""), ("sədʒi", "1", ""), ("sudʒɛ", "1", ""),
        ("sʊdʒɪ", "1", ""), ("sũn:tʃi", "1", ""),
    )),
    (39, "cloth", (
        ("mpo", "5", ""), ("mpo", "5", ""), ("mpo", "5", ""),
        ("m̩po", "5", ""), ("m̩po", "5", ""), ("m̩po", "5", ""),
        ("send̪ɾɛ", "2", ""), ("pʌt̪ai", "1", ""), ("gət̪uŋ", "4", ""),
        ("lʊgɛ", "3", ""), ("lu:ga", "3", ""),
    )),
    (40, "ring", (
        ("oɾt̪i", "1", ""), ("oɾt̪i", "1", ""), ("oɾt̪i", "1", ""),
        ("oɾt̪i", "1", ""), ("oɾt̪i", "1", ""), ("mʊd̪ɪ", "2", ""),
        ("mʊndɪ", "2", ""), ("ʊaɾt̪i", "1", ""),
        ("mun̪d̪ɪ", "2", ""), ("mʊn̪d̪ɪ", "2", ""), ("mud̪i", "2", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "20", "Printed_Page": "15",
        "Column": column, "Manual_Transcription": form,
        "Similarity_Groups": groups, "Source_Qualification": qualification,
        "Review_Status": "attested", "Confidence": "high", "Uncertainty": "",
        "Reviewer_Method": "manual visual inspection at 600 dpi; every cell rechecked in targeted 1200-dpi crops",
        "Reviewed_At": "2026-08-28", "Reviewer_Declaration": DECLARATION,
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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 55
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
