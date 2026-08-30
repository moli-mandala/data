#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 1-5.

Every response, similarity-group number, bracket code, and explicit blank below
was independently hand-keyed from physical PDF page 52 / printed page 45.  The
300-dpi page was the primary source and small IPA marks were checked in targeted
1200-dpi crops.  OCR, PDF text, legacy-font data, installed forms, and earlier
audits are not inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_001_005_lines.tsv"
CELLS_OUTPUT = HERE / "items_001_005_cells.tsv"
DECLARATION = (
    "hand-keyed-from-rendered-source; "
    "OCR-PDF-text-legacy-not-copied-or-used-to-verify"
)
METHOD = (
    "manual visual inspection of the 300-dpi rendered page; "
    "small IPA marks rechecked in targeted 1200-dpi crops"
)
SITE_CODES = tuple("0abcdefghijklmnop")
LINE_FIELDS = [
    "Line_ID", "Item", "Gloss", "PDF_Page", "Printed_Page", "Column",
    "Line_Order", "Similarity_Group", "Manual_Transcription", "Bracket_Codes",
    "Printed_Status_Text", "Line_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
CELL_FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Identity", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Similarity_Groups", "Source_Line_IDs",
    "Source_Qualification", "Review_Status", "Scope", "Confidence",
    "Uncertainty", "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]

# One tuple per printed response line, in source order:
# (item, gloss, column, similarity group, diplomatic response, bracket codes,
#  status text).  Repeated response lines are deliberately repeated.
PRINTED_LINES = [
    (1, "sky", "left", "1", "sɨlɡa", "aio", ""),
    (1, "sky", "left", "2", "ʃaɡu", "c", ""),
    (1, "sky", "left", "2", "ʃoɡa", "defghn", ""),
    (1, "sky", "left", "2", "saɡu", "b", ""),
    (1, "sky", "left", "3", "bri", "jk", ""),
    (1, "sky", "left", "4", "raŋra", "lm", ""),
    (1, "sky", "left", "5", "akaʃ", "0", ""),
    (1, "sky", "left", "6", "bniŋ", "p", ""),

    (2, "sun", "left", "1", "ʃal", "adefghino", ""),
    (2, "sun", "left", "2", "raŋʃaŋ", "m", ""),
    (2, "sun", "left", "2", "raŋʃaŋ", "l", ""),
    (2, "sun", "left", "2", "rasan", "bc", ""),
    (2, "sun", "left", "3", "jao̯ bri", "jk", ""),
    (2, "sun", "left", "4", "ʃuudʒo", "0", ""),
    (2, "sun", "left", "5", "sɨŋŋei̯", "p", ""),

    (3, "moon", "left", "1", "dʒadʒoŋ", "adefio", ""),
    (3, "moon", "left", "2", "raŋɡrɛkʼ", "c", ""),
    (3, "moon", "left", "2", "raŋrɛ", "b", ""),
    (3, "moon", "left", "3", "dʒonakʼ", "gh", ""),
    (3, "moon", "left", "3", "dʒonatʼ", "e", ""),
    (3, "moon", "left", "4", "tʰao̯ bti", "jk", ""),
    (3, "moon", "left", "5", "tʃaŋ ɨi̯", "m", ""),
    (3, "moon", "left", "5", "tʃaŋ ai̯", "l", ""),
    (3, "moon", "left", "6", "dʒanokʼ", "n", ""),
    (3, "moon", "left", "6", "dʒonakʼ", "gh", ""),
    (3, "moon", "left", "7", "tʃãd", "0", ""),
    (3, "moon", "left", "8", "bni", "p", ""),

    (4, "star", "right", "1", "aʃki", "adgho", ""),
    (4, "star", "right", "1", "askʰi", "i", ""),
    (4, "star", "right", "1", "askui̯", "lm", ""),
    (4, "star", "right", "2", "aʃikʰi", "f", ""),
    (4, "star", "right", "2", "aʃki", "adgho", ""),
    (4, "star", "right", "2", "aʃukʰi", "e", ""),
    (4, "star", "right", "2", "aʃuki", "n", ""),
    (4, "star", "right", "3", "aʃikʰi", "f", ""),
    (4, "star", "right", "3", "aʃukʰi", "e", ""),
    (4, "star", "right", "3", "askʰi", "i", ""),
    (4, "star", "right", "4", "kʰlɔr", "p", ""),
    (4, "star", "right", "4", "kʰlor", "jk", ""),
    (4, "star", "right", "5", "tara", "0bc", ""),

    (5, "cloud", "right", "0", "", "p", "no entry"),
    (5, "cloud", "right", "1", "ɡadʒɨla", "m", ""),
    (5, "cloud", "right", "1", "ɡadɨla", "ailo", ""),
    (5, "cloud", "right", "1", "ɡadɨla", "e", ""),
    (5, "cloud", "right", "1", "ɡadla", "bcdfghn", ""),
    (5, "cloud", "right", "2", "loʔo", "jk", ""),
    (5, "cloud", "right", "3", "mɛɡʰ", "0", ""),
]


def line_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    item_orders: defaultdict[int, int] = defaultdict(int)
    for item, gloss, column, group, form, codes, status_text in PRINTED_LINES:
        item_orders[item] += 1
        order = item_orders[item]
        assert codes and len(set(codes)) == len(codes)
        assert set(codes) <= set(SITE_CODES)
        status = "source_blank" if status_text == "no entry" else "attested"
        assert bool(form) == (status == "attested")
        row = {
            "Line_ID": f"i{item:03d}-l{order:02d}",
            "Item": str(item),
            "Gloss": gloss,
            "PDF_Page": "52",
            "Printed_Page": "45",
            "Column": column,
            "Line_Order": str(order),
            "Similarity_Group": group,
            "Manual_Transcription": form,
            "Bracket_Codes": codes,
            "Printed_Status_Text": status_text,
            "Line_Status": status,
            "Confidence": "high",
            "Uncertainty": "",
            "Reviewer_Method": METHOD,
            "Reviewed_At": "2026-08-29",
            "Reviewer_Declaration": DECLARATION,
        }
        assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
        rows.append(row)
    return rows


def cell_rows(lines: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    lines = line_rows() if lines is None else lines
    by_cell: defaultdict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    metadata: dict[int, tuple[str, str]] = {}
    for row in lines:
        item = int(row["Item"])
        metadata[item] = (row["Gloss"], row["Column"])
        for code in row["Bracket_Codes"]:
            by_cell[(item, code)].append(row)

    expected = {(item, code) for item in range(1, 6) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(1, 6):
        gloss, column = metadata[item]
        for code in SITE_CODES:
            source_lines = by_cell[(item, code)]
            attested = [row for row in source_lines if row["Line_Status"] == "attested"]
            blanks = [row for row in source_lines if row["Line_Status"] == "source_blank"]
            assert not (attested and blanks)
            if attested:
                form = " | ".join(row["Manual_Transcription"] for row in attested)
                groups = "|".join(row["Similarity_Group"] for row in attested)
                qualification = ""
                status = "attested"
                confidence = "high"
            else:
                assert len(blanks) == 1
                form = ""
                groups = blanks[0]["Similarity_Group"]
                qualification = 'printed "no entry"'
                status = "source_blank"
                confidence = "not_applicable"
            row = {
                "Item": str(item),
                "Gloss": gloss,
                "Site_Code": code,
                "Site_Identity": f"printed site code {code}",
                "PDF_Page": "52",
                "Printed_Page": "45",
                "Column": column,
                "Manual_Transcription": form,
                "Similarity_Groups": groups,
                "Source_Line_IDs": "|".join(line["Line_ID"] for line in source_lines),
                "Source_Qualification": qualification,
                "Review_Status": status,
                "Scope": "control_audit_only" if code == "0" else "neutral_unreconciled",
                "Confidence": confidence,
                "Uncertainty": "",
                "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-29",
                "Reviewer_Declaration": DECLARATION,
            }
            assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
            out.append(row)
    return out


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    lines = line_rows()
    cells = cell_rows(lines)
    assert len(lines) == 47
    assert len(cells) == 85
    assert sum(row["Line_Status"] == "source_blank" for row in lines) == 1
    assert sum(row["Review_Status"] == "attested" for row in cells) == 84
    assert sum(row["Review_Status"] == "source_blank" for row in cells) == 1
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 95
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
