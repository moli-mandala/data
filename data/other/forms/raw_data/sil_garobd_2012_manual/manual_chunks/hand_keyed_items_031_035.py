#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 31-35.

Every response, similarity-group number, and bracket code below was independently
hand-keyed from physical PDF page 56 / printed page 49. The 300-dpi page was the
primary source and small IPA marks were checked in targeted 1200-dpi crops. OCR,
PDF text, legacy-font data, installed forms, and earlier audits are not inputs.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_031_035_lines.tsv"
CELLS_OUTPUT = HERE / "items_031_035_cells.tsv"
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
    31: ("56", "49", "left"),
    32: ("56", "49", "left"),
    33: ("56", "49", "left"),
    34: ("56", "49", "left"),
    35: ("56", "49", "right"),
}

# One tuple per printed response line, in source order.
# Repeated site assignments in items 32 and 35 are deliberately preserved.
PRINTED_LINES = [
    (31, "evening", "1", "hantʰam", "adefo", ""),
    (31, "evening", "1", "hɛntʰam", "ghi", ""),
    (31, "evening", "2", "gasam", "lm", ""),
    (31, "evening", "2", "gasum", "bc", ""),
    (31, "evening", "3", "motʼ", "jk", ""),
    (31, "evening", "4", "duʃumi", "n", ""),
    (31, "evening", "5", "ʃondʰa", "0", ""),
    (31, "evening", "6", "ɖʒanmot", "p", ""),

    (32, "night", "1", "wal", "adfghlmno", ""),
    (32, "night", "1", "wala", "e", ""),
    (32, "night", "2", "pʰarokʼ", "bc", ""),
    (32, "night", "3", "tʃatʃu", "jk", ""),
    (32, "night", "4", "wala", "e", ""),
    (32, "night", "4", "wallo", "i", ""),
    (32, "night", "5", "rat", "0", ""),
    (32, "night", "6", "mot", "p", ""),

    (33, "paddy rice", "1", "mi", "adghino", ""),
    (33, "paddy rice", "2", "mai̯", "bceflm", ""),
    (33, "paddy rice", "3", "ɖʒiba", "jk", ""),
    (33, "paddy rice", "3", "ɖʒiba", "p", ""),
    (33, "paddy rice", "4", "dʰan", "0", ""),

    (34, "uncooked rice", "1", "mai̯ruŋ", "bcefl", ""),
    (34, "uncooked rice", "1", "mai̯roŋ", "m", ""),
    (34, "uncooked rice", "1", "mɛroŋ", "dn", ""),
    (34, "uncooked rice", "1", "miroŋ", "aghio", ""),
    (34, "uncooked rice", "2", "kʰao̯", "jkp", ""),
    (34, "uncooked rice", "3", "tʃal", "0", ""),

    (35, "cooked rice", "1", "mi", "adghino", ""),
    (35, "cooked rice", "2", "mai̯", "bcefm", ""),
    (35, "cooked rice", "3", "ɖʒa", "jkp", ""),
    (35, "cooked rice", "4", "mai̯mɨn", "lm", ""),
    (35, "cooked rice", "5", "bʰat", "0", ""),
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
        status = "source_blank" if status_text == "no entry" else "attested"
        assert bool(form) == (status == "attested")
        row = {
            "Line_ID": f"i{item:03d}-l{order:02d}", "Item": str(item),
            "Gloss": gloss, "PDF_Page": pdf_page, "Printed_Page": printed_page,
            "Column": column, "Line_Order": str(order), "Similarity_Group": group,
            "Manual_Transcription": form, "Bracket_Codes": codes,
            "Printed_Status_Text": status_text, "Line_Status": status,
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

    expected = {(item, code) for item in range(31, 36) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(31, 36):
        gloss, pdf_page, printed_page, column = metadata[item]
        for code in SITE_CODES:
            source_lines = by_cell[(item, code)]
            attested = [row for row in source_lines if row["Line_Status"] == "attested"]
            blanks = [row for row in source_lines if row["Line_Status"] == "source_blank"]
            assert not blanks
            form = " | ".join(row["Manual_Transcription"] for row in attested)
            groups = "|".join(row["Similarity_Group"] for row in attested)
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
    assert len(lines) == 32
    assert len(cells) == 85
    assert all(row["Review_Status"] == "attested" for row in cells)
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells) == 87
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
