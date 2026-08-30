#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 101-105.

Every response, group number, bracket code, and repetition below was
independently hand-keyed from physical PDF page 65 / printed page 58. The
300-dpi page was primary and small IPA marks were checked in targeted 1200-dpi
crops. OCR, PDF text, legacy data, installed forms, and earlier audits are not
inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_101_105_lines.tsv"
CELLS_OUTPUT = HERE / "items_101_105_cells.tsv"
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
PAGE_BY_ITEM = {
    **{item: ("65", "58", "left") for item in range(101, 103)},
    **{item: ("65", "58", "right") for item in range(103, 106)},
}

# One tuple per printed response line, in source order.
PRINTED_LINES = [
    (101, "neck", "1", "gɨtʼdok", "o", ""),
    (101, "neck", "1", "gɨtok", "i", ""),
    (101, "neck", "1", "godok", "defghn", ""),
    (101, "neck", "1", "gutok", "a", ""),
    (101, "neck", "2", "kalkʰu", "bc", ""),
    (101, "neck", "3", "kraŋ", "jk", ""),
    (101, "neck", "3", "kraŋ", "p", ""),
    (101, "neck", "3", "tokɾɛŋ", "lm", ""),
    (101, "neck", "4", "gola", "0", ""),

    (102, "hair", "1", "kʰɨnni", "adio", ""),
    (102, "hair", "2", "kʰau̯", "bclm", ""),
    (102, "hair", "3", "skʰdɨŋ", "h", ""),
    (102, "hair", "4", "snia̯k", "p", ""),
    (102, "hair", "4", "snʲɨk", "jk", ""),
    (102, "hair", "5", "tʃul", "0efgn", ""),

    (103, "eye", "1", "mɨkon", "ao", ""),
    (103, "eye", "1", "mɨkɾɛŋ", "lm", ""),
    (103, "eye", "1", "mɨkɾon", "ghi", ""),
    (103, "eye", "1", "mokʼkon", "bc", ""),
    (103, "eye", "1", "mukɾoŋ", "d", ""),
    (103, "eye", "2", "mɨkɾɛŋ", "lm", ""),
    (103, "eye", "2", "mɨkɾon", "ghi", ""),
    (103, "eye", "2", "mukɾoŋ", "d", ""),
    (103, "eye", "2", "mukuruŋ", "efn", ""),
    (103, "eye", "3", "kʰmat", "jkp", ""),
    (103, "eye", "4", "tʃok", "0", ""),

    (104, "nose", "1", "gɨŋ", "aghio", ""),
    (104, "nose", "1", "guŋ", "defn", ""),
    (104, "nose", "2", "nakʼkuŋ", "bc", ""),
    (104, "nose", "2", "nakʰuŋ", "lm", ""),
    (104, "nose", "3", "lɨmʊt", "jk", ""),
    (104, "nose", "3", "lɨmut", "p", ""),
    (104, "nose", "4", "nak", "0", ""),

    (105, "ear", "1", "natʃɨl", "adghio", ""),
    (105, "ear", "1", "natʃul", "efn", ""),
    (105, "ear", "2", "nakʰar", "bclm", ""),
    (105, "ear", "3", "lɨkur", "jkp", ""),
    (105, "ear", "4", "kan", "0", ""),
]


def line_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    item_orders: defaultdict[int, int] = defaultdict(int)
    for item, gloss, group, form, codes, status_text in PRINTED_LINES:
        item_orders[item] += 1
        order = item_orders[item]
        pdf_page, printed_page, column = PAGE_BY_ITEM[item]
        assert codes and len(set(codes)) == len(codes)
        assert set(codes) <= set(SITE_CODES)
        assert not status_text and form
        row = {
            "Line_ID": f"i{item:03d}-l{order:02d}", "Item": str(item),
            "Gloss": gloss, "PDF_Page": pdf_page, "Printed_Page": printed_page,
            "Column": column, "Line_Order": str(order), "Similarity_Group": group,
            "Manual_Transcription": form, "Bracket_Codes": codes,
            "Printed_Status_Text": status_text, "Line_Status": "attested",
            "Confidence": "high", "Uncertainty": "", "Reviewer_Method": METHOD,
            "Reviewed_At": "2026-08-29", "Reviewer_Declaration": DECLARATION,
        }
        assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
        rows.append(row)
    return rows


def cell_rows(lines: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    lines = line_rows() if lines is None else lines
    by_cell: defaultdict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    metadata: dict[int, tuple[str, str, str, str]] = {}
    for row in lines:
        item = int(row["Item"])
        metadata[item] = (row["Gloss"], row["PDF_Page"], row["Printed_Page"], row["Column"])
        for code in row["Bracket_Codes"]:
            by_cell[(item, code)].append(row)

    expected = {(item, code) for item in range(101, 106) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(101, 106):
        gloss, pdf_page, printed_page, column = metadata[item]
        for code in SITE_CODES:
            source_lines = by_cell[(item, code)]
            form = " | ".join(row["Manual_Transcription"] for row in source_lines)
            groups = "|".join(row["Similarity_Group"] for row in source_lines)
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Site_Identity": f"printed site code {code}", "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": form, "Similarity_Groups": groups,
                "Source_Line_IDs": "|".join(line["Line_ID"] for line in source_lines),
                "Source_Qualification": "", "Review_Status": "attested",
                "Scope": "control_audit_only" if code == "0" else "neutral_unreconciled",
                "Confidence": "high", "Uncertainty": "", "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-29", "Reviewer_Declaration": DECLARATION,
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
    assert len(lines) == 38
    assert len(cells) == 85
    assert all(row["Review_Status"] == "attested" for row in cells)
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells) == 91
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
