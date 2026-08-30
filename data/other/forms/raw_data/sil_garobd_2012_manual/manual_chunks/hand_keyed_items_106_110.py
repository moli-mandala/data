#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 106-110.

Every response, group number, bracket code, repetition, and explicit blank below
was independently hand-keyed from physical PDF page 66 / printed page 59. The
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
LINES_OUTPUT = HERE / "items_106_110_lines.tsv"
CELLS_OUTPUT = HERE / "items_106_110_cells.tsv"
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
    **{item: ("66", "59", "left") for item in range(106, 109)},
    **{item: ("66", "59", "right") for item in range(109, 111)},
}

# One tuple per printed response line, in source order.
PRINTED_LINES = [
    (106, "cheek", "0", "", "p", "no entry"),
    (106, "cheek", "1", "pʰai̯tʰopʼ", "ef", ""),
    (106, "cheek", "1", "pʰai̯tʰupa", "lm", ""),
    (106, "cheek", "1", "pʰitʰopʼ", "aio", ""),
    (106, "cheek", "2", "pʰai̯tʰopʼ", "ef", ""),
    (106, "cheek", "2", "pʰɛtʰɛŋ", "dg", ""),
    (106, "cheek", "2", "pʰɛtʰɪŋ", "n", ""),
    (106, "cheek", "2", "pʰitʰɨŋ", "h", ""),
    (106, "cheek", "2", "pʰitʰopʼ", "aio", ""),
    (106, "cheek", "3", "pai̯tʰokʼ", "bc", ""),
    (106, "cheek", "3", "pʰai̯tʰopʼ", "ef", ""),
    (106, "cheek", "4", "tiwai̯tʼ", "jk", ""),
    (106, "cheek", "5", "gal", "0", ""),

    (107, "chin", "1", "kʰuʔdubu", "efghn", ""),
    (107, "chin", "1", "kʰudɨhu", "d", ""),
    (107, "chin", "1", "kʰudumbok", "ai", ""),
    (107, "chin", "1", "kʰudumuk", "o", ""),
    (107, "chin", "2", "kadɨmbai̯", "lm", ""),
    (107, "chin", "2", "kʰudumbok", "ai", ""),
    (107, "chin", "3", "tɨmuʔ", "p", ""),
    (107, "chin", "4", "katʰolok", "bc", ""),
    (107, "chin", "5", "taŋbon", "jk", ""),
    (107, "chin", "6", "tʃibuk", "0", ""),

    (108, "mouth", "1", "kʰuʃukʼ", "adio", ""),
    (108, "mouth", "1", "kʰutʃukʼ", "lm", ""),
    (108, "mouth", "2", "kʰutʃar", "bc", ""),
    (108, "mouth", "2", "kʰutʃukʼ", "lm", ""),
    (108, "mouth", "3", "kʰu", "efghn", ""),
    (108, "mouth", "4", "gap", "jkp", ""),
    (108, "mouth", "5", "muk", "0", ""),

    (109, "tongue", "1", "sri", "ai", ""),
    (109, "tongue", "2", "tʰalai̯", "bc", ""),
    (109, "tongue", "3", "ʃɛlɛbakʼ", "e", ""),
    (109, "tongue", "3", "sɛribakʼ", "h", ""),
    (109, "tongue", "3", "sɛrɛbakʼ", "gn", ""),
    (109, "tongue", "3", "sribakʼ", "do", ""),
    (109, "tongue", "4", "ʃɛlabakʼ", "f", ""),
    (109, "tongue", "4", "ʃɛlɛbakʼ", "e", ""),
    (109, "tongue", "4", "sɛribakʼ", "h", ""),
    (109, "tongue", "4", "sɛrɛbakʼ", "gn", ""),
    (109, "tongue", "5", "tʰɨloi̯tʼ", "jk", ""),
    (109, "tongue", "5", "tʰloi̯t", "p", ""),
    (109, "tongue", "6", "tʰɛlampa", "lm", ""),
    (109, "tongue", "7", "dʒɪb", "0", ""),

    (110, "tooth", "1", "wa", "adefghilmno", ""),
    (110, "tooth", "2", "tʰa", "bc", ""),
    (110, "tooth", "3", "moi̯n", "jkp", ""),
    (110, "tooth", "4", "dãt", "0", ""),
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

    expected = {(item, code) for item in range(106, 111) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(106, 111):
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
    assert len(lines) == 48
    assert len(cells) == 85
    assert sum(row["Review_Status"] == "attested" for row in cells) == 84
    assert sum(row["Review_Status"] == "source_blank" for row in cells) == 1
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 99
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
