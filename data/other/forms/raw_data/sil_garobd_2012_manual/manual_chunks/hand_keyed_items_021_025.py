#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 21-25.

Every response, similarity-group number, bracket code, and explicit blank below
was independently hand-keyed from physical PDF pages 54-55 / printed pages 47-48.
The 300-dpi pages were the primary source and small IPA marks were checked in
targeted 1200-dpi crops. OCR, PDF text, legacy-font data, installed forms, and
earlier audits are not inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_021_025_lines.tsv"
CELLS_OUTPUT = HERE / "items_021_025_cells.tsv"
DECLARATION = (
    "hand-keyed-from-rendered-source; "
    "OCR-PDF-text-legacy-not-copied-or-used-to-verify"
)
METHOD = (
    "manual visual inspection of the 300-dpi rendered pages; "
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
    21: ("54", "47", "right"),
    22: ("54", "47", "right"),
    23: ("55", "48", "left"),
    24: ("55", "48", "left"),
    25: ("55", "48", "left"),
}

# One tuple per printed response line, in source order:
# (item, gloss, similarity group, diplomatic response, bracket codes, status text).
PRINTED_LINES = [
    (21, "silver", "0", "", "p", "no entry"),
    (21, "silver", "1", "rupa", "0abcdefghijklmno", ""),

    (22, "today", "1", "daʔaʃal", "adfio", ""),
    (22, "today", "1", "daʔasal", "e", ""),
    (22, "today", "1", "duʔʃal", "gn", ""),
    (22, "today", "1", "duʔuʃal", "h", ""),
    (22, "today", "2", "tai̯ni", "bc", ""),
    (22, "today", "2", "tani", "m", ""),
    (22, "today", "3", "taan", "l", ""),
    (22, "today", "4", "hɨnta", "jkp", ""),
    (22, "today", "5", "adʒ", "0", ""),

    (23, "yesterday", "0", "", "l", "no entry"),
    (23, "yesterday", "1", "mɨja", "m", ""),
    (23, "yesterday", "1", "midʒa", "adefghn", ""),
    (23, "yesterday", "1", "midʒao̯", "io", ""),
    (23, "yesterday", "1", "mui̯ja", "bc", ""),
    (23, "yesterday", "2", "monao̯", "jk", ""),
    (23, "yesterday", "2", "mot nao̯", "p", ""),
    (23, "yesterday", "3", "gotokal / kalke", "0", ""),

    (24, "tomorrow", "0", "", "l", "no entry"),
    (24, "tomorrow", "1", "kʰnal", "a", ""),
    (24, "tomorrow", "2", "gɛnɛ", "bc", ""),
    (24, "tomorrow", "3", "ambɨn", "o", ""),
    (24, "tomorrow", "3", "ambɨnu", "d", ""),
    (24, "tomorrow", "3", "ambono", "e", ""),
    (24, "tomorrow", "3", "ambuno", "i", ""),
    (24, "tomorrow", "4", "kʰinapo", "f", ""),
    (24, "tomorrow", "5", "kalu", "gh", ""),
    (24, "tomorrow", "6", "rɨtip", "p", ""),
    (24, "tomorrow", "7", "hanɛp", "m", ""),
    (24, "tomorrow", "8", "ɖʒɛlo", "n", ""),
    (24, "tomorrow", "9", "agamikal", "0", ""),
    (24, "tomorrow", "A", "ɨrtip", "jk", ""),

    (25, "week", "0", "", "a", "no entry"),
    (25, "week", "1", "hatai̯", "bc", ""),
    (25, "week", "1", "hatʰai̯", "e", ""),
    (25, "week", "2", "ʃalsni", "i", ""),
    (25, "week", "3", "ju", "j", ""),
    (25, "week", "4", "tiu̯", "kp", ""),
    (25, "week", "5", "anti", "o", ""),
    (25, "week", "6", "ʃɔpta", "0dfghlmn", ""),
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
            "Line_ID": f"i{item:03d}-l{order:02d}",
            "Item": str(item),
            "Gloss": gloss,
            "PDF_Page": pdf_page,
            "Printed_Page": printed_page,
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
    metadata: dict[int, tuple[str, str, str, str]] = {}
    for row in lines:
        item = int(row["Item"])
        metadata[item] = (row["Gloss"], row["PDF_Page"], row["Printed_Page"], row["Column"])
        for code in row["Bracket_Codes"]:
            by_cell[(item, code)].append(row)

    expected = {(item, code) for item in range(21, 26) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(21, 26):
        gloss, pdf_page, printed_page, column = metadata[item]
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
                "PDF_Page": pdf_page,
                "Printed_Page": printed_page,
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
    assert len(lines) == 41
    assert len(cells) == 85
    assert sum(row["Review_Status"] == "attested" for row in cells) == 81
    assert sum(row["Review_Status"] == "source_blank" for row in cells) == 4
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 81
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
