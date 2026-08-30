#!/usr/bin/env python3
"""Write the OCR-blind Bonda 2022-005 ledger for physical p. 15.

Every response, similarity-group label, and qualifier below was independently
hand-keyed from 600-dpi and 1200-dpi rendered source images. PDF extraction and
OCR are not inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "items_001_005_hand_keyed.tsv"
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

# Each cell is (manual transcription, printed similarity groups, qualifier).
# ` | ` preserves multiple explicitly printed responses within one conceptual
# cell; repeated forms remain repeated because the source prints both.
PAGE_DECISIONS = [
    (1, "body", (
        ("ɲeɾi", "2", ""), ("ɲeɾi", "2", ""), ("ɛɾi", "2", ""),
        ("ɛɾi", "2", ""), ("ɛɾi", "2", ""), ("gagoɖe", "1", ""),
        ("nɛɾi", "2", ""), ("gagəɖe", "1", ""), ("uɾɛ", "4", ""),
        ("gɛgɔɽ", "1", ""), ("soɾiɾo", "3", ""),
    )),
    (2, "head", (
        ("bob", "3", ""), ("bo:b'", "3", ""), ("bo:b'", "3", ""),
        ("bo:b", "3", ""), ("bo:", "3", ""), ("bo:b", "3", ""),
        ("bob", "3", ""), ("bhaha", "2", ""), ("bɛ", "2", ""),
        ("muɳɖ", "1", ""), ("mũnɖɔ", "1", ""),
    )),
    (3, "hair", (
        ("ʊgʔbob' | ɭuibo", "5|6", "second response is printed with qualifier '(body hair)'"),
        ("ʊgʔbob' | ɭuibo", "5|6", "second response is printed with qualifier '(body hair)'"),
        ("ʊgʔbob' | ɭuibo", "5|6", "second response is printed with qualifier '(body hair)'"),
        ("ʊgʔbo | ʊgʔbo", "2|5", "source prints the same response twice with different similarity groups"),
        ("ʊkʔbo | ʊkʔbo", "2|5", "source prints the same response twice with different similarity groups"),
        ("ʊkʔbo | ʊkʔbo", "2|5", "source prints the same response twice with different similarity groups"),
        ("seɳɖi", "1", ""), ("ʊbɔ", "2", ""), ("t̪ɪkuɪ", "4", ""),
        ("tʃeɳɖɪ", "1", ""), ("balə", "3", ""),
    )),
    (4, "face", (
        ("sʌɾmo", "1", ""), ("saɾmo", "1", ""), ("sʌɾmo", "1", ""),
        ("sʌɾmo", "1", ""), ("sʌɾmo", "1", ""), ("sʌɾmo", "1", ""),
        ("səɾmo", "1", ""), ("saɾmua", "1", ""), ("mokʌm", "2", ""),
        ("mʊ̃", "3", ""), ("mũhə", "3", ""),
    )),
    (5, "eye", (
        ("mo", "2", ""), ("mo", "2", ""), ("mo", "2", ""),
        ("mo:", "2", ""), ("mʊo", "2", ""), ("mo:", "2", ""),
        ("mo", "2", ""), ("mʊa", "2", ""), ("mɐn", "3", ""),
        ("ɐ̃kɪ", "1", ""), ("ɑkhi", "1", ""),
    )),
]


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item, gloss, cells in PAGE_DECISIONS:
        assert len(cells) == len(SITES) == 11
        for (site_code, site_name, column), (form, groups, qualification) in zip(SITES, cells, strict=True):
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": site_code,
                "Site_Name": site_name, "PDF_Page": "15", "Printed_Page": "10",
                "Column": column, "Manual_Transcription": form,
                "Similarity_Groups": groups, "Source_Qualification": qualification,
                "Review_Status": "attested", "Confidence": "high", "Uncertainty": "",
                "Reviewer_Method": "manual visual inspection at 600 dpi; every cell rechecked at 1200 dpi",
                "Reviewed_At": "2026-08-28", "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            out.append(row)
    return out


def main() -> None:
    output_rows = rows()
    assert len(output_rows) == 55
    assert len({(row["Item"], row["Site_Code"]) for row in output_rows}) == 55
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in output_rows) == 61
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} hand-keyed cells to {OUTPUT}")


if __name__ == "__main__":
    main()
