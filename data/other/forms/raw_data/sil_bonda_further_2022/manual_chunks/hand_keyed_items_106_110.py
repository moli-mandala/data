#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for items 106-110.

Every conceptual cell was independently reviewed from rendered source pages
at 600 dpi and rechecked in targeted 1200-dpi crops. OCR, PDF text, and
prior-source readings are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_106_110_hand_keyed.tsv"
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
    (106, "mother", (
        ("dʒoŋ", "4", ""), ("joŋ", "1", ""),
        ("joŋ", "1", ""), ("ijoŋ", "1", ""),
        ("ijoŋ", "1", ""), ("ijoŋ", "1", ""),
        ("ijoŋ", "1", ""), ("ijʌŋ", "1", ""),
        ("ɛjɛ", "2", ""), ("ɛjɛ", "2", ""),
        ("mɑʔ", "3", ""),
    )),
    (107, "older brother", (
        ("maŋ", "5", ""), ("maŋ", "5", ""),
        ("maŋ", "5", ""), ("maŋ", "5", ""),
        ("mɳa maŋ", "6", ""), ("maŋ", "5", ""),
        ("moɖo bujɛŋ", "2", ""), ("mɳane naŋ", "6", ""),
        ("ɖɛɖɛ", "3", ""), ("nɛnɛ", "4", ""),
        ("nõnɑʔ", "4", ""),
    )),
    (108, "younger brother", (
        ("me", "9", ""), ("me", "9", ""),
        ("me", "9", ""),
        ("biaŋ | meʔ", "6|6", "two responses printed comma-separated on the group-6 line for KAD"),
        ("ɖau maŋ", "8", ""),
        ("biaŋ | meʔ", "6|6", "two responses printed comma-separated on the group-6 line for RAS"),
        ("mijen bʊjɛŋ | mijenbɛi", "2|2", "two responses printed on separate group-2 lines for GUT"),
        ("ɖhabõja | ɖhanepe", "1|7", "two responses printed on separate group-1 and group-7 lines for BIA"),
        ("ʌɳɛ", "3", ""), ("pila", "4", ""),
        ("tʃhoʈabɦɑi", "5", ""),
    )),
    (109, "older sister", (
        ("mɪ", "7", ""), ("miŋ", "7", ""),
        ("miŋ", "7", ""), ("miŋ", "7", ""),
        ("miŋ", "7", ""), ("miŋ", "7", ""),
        ("moɖo mimiŋ", "2", ""), ("miŋ", "7", ""),
        ("ʌbinom", "3", ""), ("ɛpɛ", "4", ""),
        ("nɑn:i | ɖiɖi", "5|6", "two responses printed on separate group-5 and group-6 lines for ODI"),
    )),
    (110, "younger sister", (
        ("kʊɪ", "8", ""), ("kʊɪ", "8", ""),
        ("kʊɪ", "8", ""), ("t̪ʊɳɑ", "5", ""),
        ("t̪ʊɳɑ", "5", ""), ("t̪ʊɳɑ", "5", ""),
        ("mijent̪onen | mijen boini", "2|3", "two responses printed on separate group-2 and group-3 lines for GUT"),
        ("ɖhanet̪həɳɑ", "1", ""), ("guɳɪ", "3", ""),
        ("nʊnɪ", "3", ""), ("sənəbɦouni", "4", ""),
    )),
]


def make_row(item: int, gloss: str, site, form: str, groups: str,
             qualification: str) -> dict[str, str]:
    site_code, site_name, column = site
    row = {
        "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
        "Site_Name": site_name, "PDF_Page": "31", "Printed_Page": "26",
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
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 61
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-reviewed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
