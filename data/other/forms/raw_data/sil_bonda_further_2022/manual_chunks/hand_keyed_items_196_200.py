#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 196-200.

Every conceptual cell was independently reviewed from rendered source page 45
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_196_200_hand_keyed.tsv"
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
    (196, "run!, he ran", (
        ("ur", "7", "", "attested"),
        ("ur", "7", "", "attested"),
        ("ur", "7", "", "attested"),
        ("ur", "7", "", "attested"),
        ("ur", "7", "", "attested"),
        ("ur", "7", "", "attested"),
        ("mɛid̪ʊŋgu | d̪ʊe", "2|3", "two responses printed on separate group-2 and group-3 lines for the same two-line GUT label", "attested"),
        ("ɪr", "1", "", "attested"),
        ("jo", "4", "", "attested"),
        ("se pelɛjlɛ", "5", "", "attested"),
        ("doudibɑ", "6", "", "attested"),
    )),
    (197, "go!, he went", (
        ("ujɑ", "3", "", "attested"),
        ("ujɑ", "3", "", "attested"),
        ("jɑ", "3", "", "attested"),
        ("jɛ", "3", "", "attested"),
        ("jɛ", "3", "", "attested"),
        ("jɛ", "3", "", "attested"),
        ("mɛiʋidʒi | jɛ", "2|3", "two responses printed on separate group-2 and group-3 lines for the same two-line GUT label", "attested"),
        ("veglɑ", "1", "", "attested"),
        ("jɛ", "3", "", "attested"),
        ("se gɛlɛ", "4", "", "attested"),
        ("dʒibɑ", "5", "", "attested"),
    )),
    (198, "come!, he came", (
        ("lo", "2", "", "attested"),
        ("lo", "2", "", "attested"),
        ("lobe", "2", "", "attested"),
        ("lo:", "2", "", "attested"),
        ("lo:", "2", "", "attested"),
        ("lo:", "2", "", "attested"),
        ("mɛipiŋgi | olo", "1|2", "two responses printed on separate group-1 and group-2 lines for the same two-line GUT label", "attested"),
        ("elɑ", "6", "", "attested"),
        ("bɛɪ", "3", "", "attested"),
        ("seɛjlɛ", "4", "", "attested"),
        ("ɑ:so", "5", "", "attested"),
    )),
    (199, "speak!, he spoke", (
        ("sũ", "2", "", "attested"),
        ("sũ", "2", "", "attested"),
        ("sũ", "2", "", "attested"),
        ("suŋ", "2", "", "attested"),
        ("sũ:", "2", "", "attested"),
        ("suŋ", "2", "", "attested"),
        ("mɛisun:o | sun", "2|2", "two group-2 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("bɑrsoŋ", "1", "", "attested"),
        ("dʒelu", "3", "", "attested"),
        ("se kɔjlɛ", "4", "", "attested"),
        ("kɔhilɑ | kuhɑ", "4|4", "two comma-separated group-4 responses for the paired prompt", "attested"),
    )),
    (200, "listen!, he heard", (
        ("õŋ", "3", "", "attested"),
        ("õŋ", "3", "", "attested"),
        ("o:ŋ", "3", "", "attested"),
        ("o:ŋ", "3", "", "attested"),
        ("o:ŋ", "3", "", "attested"),
        ("o:ŋ", "3", "", "attested"),
        ("mɛioʔoø | oŋ", "2|3", "two responses printed on separate group-2 and group-3 lines for the same two-line GUT label", "attested"),
        ("nahot̪e", "6", "", "attested"),
        ("d̪ɛr", "4", "", "attested"),
        ("se sʊnlɛ", "5", "", "attested"),
        ("suno", "5", "", "attested"),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str, status: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "45", "Printed_Page": "40",
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
