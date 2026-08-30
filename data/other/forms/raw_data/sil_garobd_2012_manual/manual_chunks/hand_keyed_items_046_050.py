#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 46-50.

Every response, group number, and bracket code below was independently
hand-keyed from physical PDF pages 57-58 / printed pages 50-51. The 300-dpi
pages were primary and small IPA marks were checked in targeted 1200-dpi
crops. OCR, PDF text, legacy data, installed forms, and earlier audits are not
inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_046_050_lines.tsv"
CELLS_OUTPUT = HERE / "items_046_050_cells.tsv"
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
    **{item: ("57", "50", "right") for item in range(46, 50)},
    50: ("58", "51", "left"),
}

# One tuple per printed response line, in source order. Item 47's repeated
# assignments to sites f, l, and m are deliberately preserved.
PRINTED_LINES = [
    (46, "thorn", "1", "buʃu", "ao", ""),
    (46, "thorn", "2", "kanta", "bcdefghin", ""),
    (46, "thorn", "3", "tʃi", "jk", ""),
    (46, "thorn", "4", "asu", "lm", ""),
    (46, "thorn", "5", "kaʈa", "0", ""),
    (46, "thorn", "6", "tʃiʔ", "p", ""),

    (47, "root", "1", "ɖʒaʔdɨl", "adeghino", ""),
    (47, "root", "1", "tʃaʔdɨl", "f", ""),
    (47, "root", "1", "tʃadɨl", "lm", ""),
    (47, "root", "2", "tʃaʔdɨl", "f", ""),
    (47, "root", "2", "tʃadɨl", "lm", ""),
    (47, "root", "2", "tʃatal", "b", ""),
    (47, "root", "2", "tʃatʰal", "c", ""),
    (47, "root", "3", "tʰɔt", "p", ""),
    (47, "root", "3", "tʰɔtʼ", "jk", ""),
    (47, "root", "4", "mul", "0", ""),

    (48, "bamboo", "1", "wa", "abcdefghilno", ""),
    (48, "bamboo", "2", "sba", "jk", ""),
    (48, "bamboo", "3", "wakai̯", "m", ""),
    (48, "bamboo", "4", "bãʃ", "0", ""),
    (48, "bamboo", "5", "rɨ si", "p", ""),

    (49, "fruit", "1", "bɛtʰei̯", "n", ""),
    (49, "fruit", "1", "bitʰai̯", "ef", ""),
    (49, "fruit", "1", "bitʰi", "adghio", ""),
    (49, "fruit", "2", "tʰai̯", "bclm", ""),
    (49, "fruit", "3", "soʔ", "jk", ""),
    (49, "fruit", "3", "suʔ", "p", ""),
    (49, "fruit", "4", "pʰɔl", "0", ""),

    (50, "jackfruit", "1", "tʰai̯ʔbroŋ", "ef", ""),
    (50, "jackfruit", "1", "tʰɛʔbroŋ", "dhn", ""),
    (50, "jackfruit", "1", "tʰiʔbroŋ", "agio", ""),
    (50, "jackfruit", "2", "pantʃuŋ", "bc", ""),
    (50, "jackfruit", "2", "pantʃum", "lm", ""),
    (50, "jackfruit", "3", "soʔram", "jk", ""),
    (50, "jackfruit", "3", "su ə rəm", "p", ""),
    (50, "jackfruit", "4", "kaʈʰal", "0", ""),
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

    expected = {(item, code) for item in range(46, 51) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(46, 51):
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
    assert len(lines) == 36
    assert len(cells) == 85
    assert all(row["Review_Status"] == "attested" for row in cells)
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells) == 88
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
