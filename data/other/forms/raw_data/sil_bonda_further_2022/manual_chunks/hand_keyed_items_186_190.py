#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 186-190.

Every conceptual cell was independently reviewed from rendered source pages
43-44 at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_186_190_hand_keyed.tsv"
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
    (186, "he is, he was thirsty", (
        ("uʔd̪ad̪ʊsʊga", "6", "", "attested"),
        ("uʔd̪ad̪ʊsʊgut̪a", "6", "", "attested"),
        ("d̪aʔ id̪ʊsʊgut̪a", "6", "", "attested"),
        ("amaiud̪aʔd̪ʊsʊgani", "2", "", "attested"),
        ("ud̪aʔd̪ʊsʊgai", "2", "", "attested"),
        ("amaiud̪aʔd̪ʊsʊgani", "2", "", "attested"),
        ("mɛisosləgəigʊd̪ʊgu | sos", "3|3", "two group-3 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("ugd̪ed̪od̪ɪn̪t̪e", "1", "", "attested"),
        ("ʌd̪e", "2", "", "attested"),
        ("t̪eke sɔs kɔrt̪e rɔjlɛ", "3", "", "attested"),
        ("soso helɑʔ", "3", "", "attested"),
    )),
    (187, "sleep!, he slept", (
        ("lemo", "6", "", "attested"),
        ("lemo", "6", "", "attested"),
        ("leʔmod̪a", "6", "", "attested"),
        ("lemod̪a", "6", "", "attested"),
        ("lemod̪a", "6", "", "attested"),
        ("d̪rɪt̪a", "5", "", "attested"),
        ("d̪ud̪i | mɛid̪ud̪igu", "1|1", "two group-1 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("d̪ulaik | d̪ulaige", "1|1", "two comma-separated group-1 responses for the paired source prompt", "attested"),
        ("d̪ɪma", "2", "", "attested"),
        ("sesɔjlɛ", "3", "", "attested"),
        ("nido", "4", "", "attested"),
    )),
    (188, "lie down!, he lay down", (
        ("d̪riga", "5", "", "attested"),
        ("d̪riga", "5", "", "attested"),
        ("d̪riga", "5", "", "attested"),
        ("dʒokt̪o lemod̪a", "6", "", "attested"),
        ("dʒokt̪od̪riga", "5", "", "attested"),
        ("dʒokt̪od̪rɪt̪a", "5", "", "attested"),
        ("d̪ud̪i | mɛid̪ud̪igu", "2|2", "two group-2 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("t̪ubod̪ulaik", "1", "", "attested"),
        ("gɔnɛ", "3", "", "attested"),
        ("sed̪ul:ɛ", "1", "", "attested"),
        ("porigola", "4", "", "attested"),
    )),
    (189, "sit down!, he sat down", (
        ("laja", "2", "", "attested"),
        ("laʔja", "2", "", "attested"),
        ("leʔja", "2", "", "attested"),
        ("leʔsɛ", "2", "", "attested"),
        ("leʔsɛ", "2", "", "attested"),
        ("leʔsɛ", "2", "", "attested"),
        ("leisɛ | mɛileigɪ", "2|2", "two group-2 responses printed on consecutive lines for the same two-line GUT label", "attested"),
        ("kola | koke", "1|1", "two comma-separated group-1 responses for the paired source prompt", "attested"),
        ("kuku", "1", "", "attested"),
        ("sebɔslɛ", "3", "", "attested"),
        ("bosibɑ", "3", "", "attested"),
    )),
    (190, "give!, he gave", (
        ("be", "2", "", "attested"),
        ("be", "2", "", "attested"),
        ("be", "2", "", "attested"),
        ("beʔ", "2", "", "attested"),
        ("beʔ", "2", "", "attested"),
        ("beʔ", "2", "", "attested"),
        ("ɪn̪d̪e | mɛibed̪o", "1|2", "two responses printed on separate group-1 and group-2 lines for the same two-line GUT label", "attested"),
        ("n̪d̪e | me bike", "1|5", "two responses printed on separate group-1 and group-5 lines for BIA", "attested"),
        ("t̪eɪ", "3", "", "attested"),
        ("se d̪elɛ", "4", "", "attested"),
        ("d̪eba", "1", "", "attested"),
    )),
]


def source_page(item: int, site_code: str) -> tuple[str, str]:
    if item <= 189 or (item == 190 and site_code in {"POD", "BON"}):
        return "43", "38"
    return "44", "39"


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
    ) == 63
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
