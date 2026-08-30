#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 56-60.

Every response, group number, bracket code, and explicit blank below was
independently hand-keyed from physical PDF page 59 / printed page 52. The
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
LINES_OUTPUT = HERE / "items_056_060_lines.tsv"
CELLS_OUTPUT = HERE / "items_056_060_cells.tsv"
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
    **{item: ("59", "52", "left") for item in range(56, 59)},
    **{item: ("59", "52", "right") for item in range(59, 61)},
}

# One tuple per printed response line, in source order. Repeated assignments in
# items 56 and 57 are deliberately preserved.
PRINTED_LINES = [
    (56, "sugarcane", "0", "", "l", "no entry"),
    (56, "sugarcane", "1", "girɨtʼ", "d", ""),
    (56, "sugarcane", "1", "gorutʼ", "efghn", ""),
    (56, "sugarcane", "1", "grɨtʼ", "aio", ""),
    (56, "sugarcane", "1", "kʰrui̯tʼ", "jk", ""),
    (56, "sugarcane", "2", "girɨtʼ", "d", ""),
    (56, "sugarcane", "2", "golotʼ", "m", ""),
    (56, "sugarcane", "2", "gorutʼ", "efghn", ""),
    (56, "sugarcane", "2", "grɨtʼ", "aio", ""),
    (56, "sugarcane", "3", "kosɛr", "bc", ""),
    (56, "sugarcane", "4", "akʰ", "0", ""),
    (56, "sugarcane", "5", "kʰlui̯t", "p", ""),
    (56, "sugarcane", "5", "kʰrui̯tʼ", "jk", ""),
    (56, "sugarcane", "6", "golotʼ", "m", ""),
    (56, "sugarcane", "6", "kʰlui̯t", "p", ""),

    (57, "betelnut", "1", "gawai̯", "lm", ""),
    (57, "betelnut", "1", "goja", "bc", ""),
    (57, "betelnut", "1", "guwa", "adio", ""),
    (57, "betelnut", "1", "guwai̯", "ef", ""),
    (57, "betelnut", "2", "gui", "ghn", ""),
    (57, "betelnut", "2", "kui", "jkp", ""),
    (57, "betelnut", "3", "ʃupari", "0", ""),
    (57, "betelnut", "5", "goja", "bc", ""),
    (57, "betelnut", "5", "gui", "ghn", ""),
    (57, "betelnut", "5", "guwa", "adio", ""),
    (57, "betelnut", "5", "guwai̯", "ef", ""),

    (58, "lime for betelnut", "1", "tʃun", "0adefghijklmnop", ""),
    (58, "lime for betelnut", "1", "tʃunu", "bc", ""),

    (59, "liquor", "1", "tʃʰu", "aefo", ""),
    (59, "liquor", "2", "tʃɨu̯", "lm", ""),
    (59, "liquor", "2", "tʃu", "dghin", ""),
    (59, "liquor", "3", "mɛra", "bc", ""),
    (59, "liquor", "4", "kɛt", "jk", ""),
    (59, "liquor", "4", "kia̯t", "p", ""),
    (59, "liquor", "5", "mɔd", "0", ""),

    (60, "milk", "0", "", "p", "no entry"),
    (60, "milk", "1", "hɨmbu", "jk", ""),
    (60, "milk", "2", "dudʰ", "0abcdefghilmno", ""),
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

    expected = {(item, code) for item in range(56, 61) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(56, 61):
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
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Site_Identity": f"printed site code {code}", "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": form, "Similarity_Groups": groups,
                "Source_Line_IDs": "|".join(line["Line_ID"] for line in source_lines),
                "Source_Qualification": qualification, "Review_Status": status,
                "Scope": "control_audit_only" if code == "0" else "neutral_unreconciled",
                "Confidence": confidence, "Uncertainty": "", "Reviewer_Method": METHOD,
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
    assert sum(row["Review_Status"] == "attested" for row in cells) == 83
    assert sum(row["Review_Status"] == "source_blank" for row in cells) == 2
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 107
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
