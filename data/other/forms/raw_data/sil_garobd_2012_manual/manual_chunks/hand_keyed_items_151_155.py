#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 151-155.

Every response, group number, bracket code, repetition, and the whole-item
"[not used]" disposition below was independently hand-keyed from physical PDF
page 72 / printed page 65. The 300-dpi page was primary and small IPA marks were
checked in targeted 1200-dpi crops. OCR, PDF text, legacy data, installed forms,
and earlier audits are not inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_151_155_lines.tsv"
CELLS_OUTPUT = HERE / "items_151_155_cells.tsv"
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
    151: ("72", "65", "left"),
    **{item: ("72", "65", "right") for item in range(152, 156)},
}

# One tuple per printed response or whole-item disposition line, in source
# order. The all-site code string on item 152 is a mechanical representation
# of the source's whole-item scope, not a printed bracket code.
PRINTED_LINES = [
    (151, "clothing", "1", "baʔara", "defi", ""),
    (151, "clothing", "1", "bara", "am", ""),
    (151, "clothing", "2", "soka", "bc", ""),
    (151, "clothing", "3", "tʃʰɨnna", "gh", ""),
    (151, "clothing", "3", "tʃinna", "n", ""),
    (151, "clothing", "4", "d͜ʒai̯n", "jk", ""),
    (151, "clothing", "5", "tʃola", "l", ""),
    (151, "clothing", "6", "tʃʰolabara", "o", ""),
    (151, "clothing", "7", "poʃak", "0", ""),
    (151, "clothing", "8", "d͜ʒai̯n pʰoŋ", "p", ""),

    (152, "cloth", "", "", "0abcdefghijklmnop", "[not used]"),

    (153, "medicine", "1", "ʃam", "aefhino", ""),
    (153, "medicine", "1", "sam", "dglm", ""),
    (153, "medicine", "2", "pantʃakʼ", "bc", ""),
    (153, "medicine", "3", "luhur", "jk", ""),
    (153, "medicine", "4", "oʃud", "0", ""),
    (153, "medicine", "5", "duwai̯", "p", ""),

    (154, "paper", "1", "lɛkʼkʰa", "abcdefghiklmno", ""),
    (154, "paper", "2", "kɔt", "jp", ""),
    (154, "paper", "3", "kagod͜ʒ", "0", ""),

    (155, "needle", "1", "ʃɛlɛŋʃi", "ef", ""),
    (155, "needle", "1", "slɛŋʃi", "aio", ""),
    (155, "needle", "2", "ʃutʃʰi", "gh", ""),
    (155, "needle", "2", "ʃutʃi", "dn", ""),
    (155, "needle", "2", "suʃi", "bc", ""),
    (155, "needle", "3", "tʰɨr ria", "p", ""),
    (155, "needle", "3", "tʰiria", "jk", ""),
    (155, "needle", "4", "sɨlʃimi", "lm", ""),
    (155, "needle", "5", "ʃutʃ", "0", ""),
    (155, "needle", "5", "ʃutʃʰi", "gh", ""),
    (155, "needle", "5", "ʃutʃi", "dn", ""),
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
        if status_text == "[not used]":
            status = "not_used"
        else:
            status = "attested"
        assert bool(form) == (status == "attested")
        row = {
            "Line_ID": f"i{item:03d}-l{order:02d}", "Item": str(item),
            "Gloss": gloss, "PDF_Page": pdf_page, "Printed_Page": printed_page,
            "Column": column, "Line_Order": str(order), "Similarity_Group": group,
            "Manual_Transcription": form, "Bracket_Codes": codes,
            "Printed_Status_Text": status_text, "Line_Status": status,
            "Confidence": "high" if status == "attested" else "not_applicable",
            "Uncertainty": "", "Reviewer_Method": METHOD,
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

    expected = {(item, code) for item in range(151, 156) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(151, 156):
        gloss, pdf_page, printed_page, column = metadata[item]
        for code in SITE_CODES:
            source_lines = by_cell[(item, code)]
            attested = [row for row in source_lines if row["Line_Status"] == "attested"]
            not_used = [row for row in source_lines if row["Line_Status"] == "not_used"]
            assert not (attested and not_used)
            if attested:
                form = " | ".join(row["Manual_Transcription"] for row in attested)
                groups = "|".join(row["Similarity_Group"] for row in attested)
                qualification = ""
                status = "attested"
                confidence = "high"
            else:
                assert len(not_used) == 1
                form = ""
                groups = ""
                qualification = 'printed "[not used]" for whole item'
                status = "not_used"
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
    assert len(lines) == 31
    assert len(cells) == 85
    assert sum(row["Review_Status"] == "attested" for row in cells) == 68
    assert sum(row["Review_Status"] == "not_used" for row in cells) == 17
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 72
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed response/disposition lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
