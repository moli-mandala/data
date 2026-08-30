#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 121-125.

Every response, group number, bracket code, repetition, and explicit blank below
was independently hand-keyed from physical PDF page 68 / printed page 61. The
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
LINES_OUTPUT = HERE / "items_121_125_lines.tsv"
CELLS_OUTPUT = HERE / "items_121_125_cells.tsv"
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
    **{item: ("68", "61", "left") for item in range(121, 123)},
    **{item: ("68", "61", "right") for item in range(123, 126)},
}

# One tuple per printed response line, in source order.
PRINTED_LINES = [
    (121, "blood", "1", "hanʔtʃʰi", "efh", ""),
    (121, "blood", "1", "hantʃʰi", "adgino", ""),
    (121, "blood", "2", "tʰɨi̯", "lm", ""),
    (121, "blood", "2", "tʰi", "bc", ""),
    (121, "blood", "3", "snam", "jk", ""),
    (121, "blood", "4", "rɔkto", "0", ""),
    (121, "blood", "5", "sɨŋŋam", "p", ""),

    (122, "sweat", "0", "", "be", "no entry"),
    (122, "sweat", "1", "kʰramtʃʰi", "adfio", ""),
    (122, "sweat", "2", "ramti", "c", ""),
    (122, "sweat", "3", "ruʔutʃia", "gh", ""),
    (122, "sweat", "3", "rutʃia̯", "n", ""),
    (122, "sweat", "4", "d͜ʒɨlupʼ", "jk", ""),
    (122, "sweat", "5", "tuŋgoa̯", "m", ""),
    (122, "sweat", "5", "tuŋokʼ", "l", ""),
    (122, "sweat", "6", "gʰam", "0", ""),
    (122, "sweat", "7", "pɨruʔ", "p", ""),

    (123, "belly", "1", "bɨkʼma", "aio", ""),
    (123, "belly", "1", "bukʼma", "e", ""),
    (123, "belly", "2", "bibɨk", "h", ""),
    (123, "belly", "2", "bibukʼ", "dfgn", ""),
    (123, "belly", "2", "pipukʼ", "m", ""),
    (123, "belly", "3", "hokʼ", "bc", ""),
    (123, "belly", "4", "kʰlao̯", "jk", ""),
    (123, "belly", "5", "pɛt", "0", ""),
    (123, "belly", "6", "lau̯baʔ", "p", ""),
    (123, "belly", "7", "pipʰuʔ", "l", ""),
    (123, "belly", "7", "pipukʼ", "m", ""),

    (124, "heart (organ)", "0", "", "bp", "no entry"),
    (124, "heart (organ)", "1", "bikʰa", "ai", ""),
    (124, "heart (organ)", "1", "pikʰa", "l", ""),
    (124, "heart (organ)", "2", "pʰokʼtʃa", "c", ""),
    (124, "heart (organ)", "3", "d͜ʒaŋgi", "defghlmno", ""),
    (124, "heart (organ)", "4", "insom", "jk", ""),
    (124, "heart (organ)", "5", "ridɔi̯", "0", ""),

    (125, "back", "1", "d͜ʒaŋgɨl", "adghio", ""),
    (125, "back", "1", "d͜ʒaŋgul", "efn", ""),
    (125, "back", "2", "kundu", "bc", ""),
    (125, "back", "3", "pʰat", "p", ""),
    (125, "back", "3", "pʰatʼ", "jk", ""),
    (125, "back", "4", "kɨn", "lm", ""),
    (125, "back", "5", "pitʰ", "0", ""),
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

    expected = {(item, code) for item in range(121, 126) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(121, 126):
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
    assert len(lines) == 42
    assert len(cells) == 85
    assert sum(row["Review_Status"] == "attested" for row in cells) == 81
    assert sum(row["Review_Status"] == "source_blank" for row in cells) == 4
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 83
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
