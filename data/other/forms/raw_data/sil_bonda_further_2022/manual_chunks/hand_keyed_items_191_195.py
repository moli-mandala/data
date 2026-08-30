#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 191-195.

Every conceptual cell was independently reviewed from rendered source page 44
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_191_195_hand_keyed.tsv"
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
    (191, "it burns, it burned", (
        ("d̪ogʊt̪a", "9", "", "attested"),
        ("d̪ogʊt̪a", "9", "", "attested"),
        ("d̪ogʊt̪a", "9", "", "attested"),
        ("sʊŋod̪oʔgani", "7", "", "attested"),
        ("sʊŋod̪oʔgani", "7", "", "attested"),
        ("sʊŋod̪oʔgani", "7", "", "attested"),
        ("ləgəigʊni | mɛigɛbo", "2|3", "two responses printed on separate group-2 and group-3 lines for the same two-line GUT label", "attested"),
        ("d̪uʋad̪iŋke | tʃigmuaga", "1|1", "two comma-separated group-1 responses, with the second wrapped to the following source line, for the paired prompt", "attested"),
        ("t̪ʌbgu", "4", "", "attested"),
        ("sedʒɔjlɛgɛjlɛ", "5", "", "attested"),
        ("dʒolibɑ", "6", "", "attested"),
    )),
    (192, "don’t die!, he died", (
        ("agoiga", "1", "", "attested"),
        ("agoiga", "1", "", "attested"),
        ("agʊiga", "1", "", "attested"),
        ("agoiga", "1", "", "attested"),
        ("agoiga", "1", "", "attested"),
        ("agoiga", "1", "", "attested"),
        ("goisɛ | mɛigoigɪ | oɾ goigu", "1|1|1", "three group-1 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("agoige | me goige", "1|1", "two comma-separated group-1 responses, with the second wrapped to the following source line, for the paired prompt", "attested"),
        ("kɪ ru", "2", "", "attested"),
        ("semɔrɪgɛlɛ", "3", "", "attested"),
        ("moɾonõ", "4", "", "attested"),
    )),
    (193, "don’t kill!, he killed", (
        ("ogoi", "8", "", "attested"),
        ("ogoi", "8", "", "attested"),
        ("ogʊjo", "8", "", "attested"),
        ("abugo", "7", "", "attested"),
        ("abugo", "7", "", "attested"),
        ("abugo", "7", "", "attested"),
        ("mɛibʊo | bʊq", "2|3", "two responses printed on separate group-2 and group-3 lines for the same two-line GUT label", "attested"),
        ("aboge | me bagoige", "1|1", "two comma-separated group-1 responses, with the second wrapped to the following source line, for the paired prompt", "attested"),
        ("lɛ", "4", "", "attested"),
        ("se merɪd̪elɛ", "5", "", "attested"),
        ("maɾibɑ", "6", "", "attested"),
    )),
    (194, "fly!, it flew", (
        ("ʋalo", "1", "", "attested"),
        ("ʋalo", "1", "", "attested"),
        ("uʋala", "1", "", "attested"),
        ("ud̪ugani", "2", "", "attested"),
        ("ud̪ugani", "2", "", "attested"),
        ("ud̪ugani", "2", "", "attested"),
        ("mɛiʋd̪eigʊ | ʊd̪ei", "2|2", "two group-2 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("t̪hulie", "3", "", "attested"),
        ("ud̪eɪ", "2", "", "attested"),
        ("ɛt̪eurlɛ", "2", "", "attested"),
        ("udutʃi", "2", "", "attested"),
    )),
    (195, "walk!, he walked", (
        ("uriŋ", "3", "", "attested"),
        ("uriŋ", "3", "", "attested"),
        ("uriŋ", "3", "", "attested"),
        ("uriŋ", "3", "", "attested"),
        ("uriŋ", "3", "", "attested"),
        ("uriŋ", "3", "", "attested"),
        ("əŋsʊŋ:ɛ | mɛiəŋsʊŋgu", "2|2", "two group-2 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("olaiŋ", "6", "", "attested"),
        ("od̪ɪŋ", "3", "", "attested"),
        ("seɪnd̪lɛ", "4", "", "attested"),
        ("tʃalibɑʔ", "5", "", "attested"),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str, status: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "44", "Printed_Page": "39",
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
    ) == 64
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
