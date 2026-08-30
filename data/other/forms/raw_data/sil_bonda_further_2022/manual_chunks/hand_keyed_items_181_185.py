#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 181-185.

Every conceptual cell was independently reviewed from rendered source pages
42-43 at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_181_185_hand_keyed.tsv"
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
    (181, "all", (
        ("gulai", "3", "", "attested"),
        ("gulai", "3", "", "attested"),
        ("gulai", "3", "", "attested"),
        ("gulai", "3", "", "attested"),
        ("gulai", "3", "", "attested"),
        ("gulai", "3", "", "attested"),
        ("səb:u", "2", "", "attested"),
        ("t̪hʌnd̪e", "4", "", "attested"),
        ("sɔbu", "2", "", "attested"),
        ("sɔbʊ", "2", "", "attested"),
        ("sobu", "2", "", "attested"),
    )),
    (182, "eat!, he ate", (
        ("sʊm", "2", "", "attested"),
        ("sʊm", "2", "", "attested"),
        ("sʊm", "2", "", "attested"),
        ("sum", "2", "", "attested"),
        ("sum", "2", "", "attested"),
        ("sum", "2", "", "attested"),
        ("mɛisomo | som", "2|2", "two group-2 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("tʃoŋ", "1", "", "attested"),
        ("gɛ", "3", "", "attested"),
        ("sekɛjlɛ", "4", "", "attested"),
        ("kaibə", "5", "", "attested"),
    )),
    (183, "bite!, he bit", (
        ("o:b", "2", "", "attested"),
        ("o:b", "2", "", "attested"),
        ("oʔp", "2", "", "attested"),
        ("ɔʔɔp", "2", "", "attested"),
        ("ɔʔɔb", "2", "", "attested"),
        ("ɔʔɔp", "2", "", "attested"),
        ("ob", "2", "", "attested"),
        ("ha", "1", "", "attested"),
        ("lom", "3", "", "attested"),
        ("setʃɛblɛ", "4", "", "attested"),
        ("tsubaila", "5", "", "attested"),
    )),
    (184, "he is, he was hungry", (
        ("kʊd̪ʊgʊt̪a", "2", "", "attested"),
        ("kʊd̪ʊgʊt̪a", "2", "", "attested"),
        ("kʊd̪ʊgʊt̪a", "2", "", "attested"),
        ("kud̪ugaʔani", "2", "", "attested"),
        ("kud̪uga", "2", "", "attested"),
        ("kud̪ugaʔani", "2", "", "attested"),
        ("kʊd̪ʊgʊni | kʊd̪ʊgud̪ʊg:u", "2|2", "two group-2 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("kɪd̪esod̪ɪn̪t̪e", "1", "", "attested"),
        ("but̪", "4", "", "attested"),
        ("t̪eke buk kɔrt̪e rɔjlɛ", "5", "", "attested"),
        ("bɦoko helɑʔ", "6", "", "attested"),
    )),
    (185, "drink!, he drank", (
        ("uʔ", "1", "", "attested"),
        ("uʔ", "1", "", "attested"),
        ("i", "7", "", "attested"),
        ("uː", "1", "", "attested"),
        ("uː", "1", "", "attested"),
        ("ʊd̪o", "2", "", "attested"),
        ("mɛid̪o | it̪unɪŋ", "2|3", "two responses printed on separate group-2 and group-3 lines for the same two-line GUT label", "attested"),
        ("uk̚ | me uke", "1|1", "two comma-separated group-1 responses for the paired source prompt", "attested"),
        ("gɪd̪e", "4", "", "attested"),
        ("se kɛjlɛ", "5", "", "attested"),
        ("piːbɑ", "6", "", "attested"),
    )),
]


def source_page(item: int, site_code: str) -> tuple[str, str]:
    if item <= 182 or (item == 183 and site_code not in {"RON", "ODI"}):
        return "42", "37"
    return "43", "38"


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
    ) == 59
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
