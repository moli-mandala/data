#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 76-80.

Every response, group number, bracket code, and repetition below was
independently hand-keyed from physical PDF pages 61-62 / printed pages 54-55.
The 300-dpi pages were primary and small IPA marks were checked in targeted
1200-dpi crops. OCR, PDF text, legacy data, installed forms, and earlier audits
are not inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_076_080_lines.tsv"
CELLS_OUTPUT = HERE / "items_076_080_cells.tsv"
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
    76: ("61", "54", "right"),
    **{item: ("62", "55", "left") for item in range(77, 81)},
}

# One tuple per printed response line, in source order.
PRINTED_LINES = [
    (76, "turtle", "1", "tʃapʼpa", "ad", ""),
    (76, "turtle", "1", "tʃɛpʼpa", "i", ""),
    (76, "turtle", "2", "kaʃiŋ", "bc", ""),
    (76, "turtle", "3", "tʃid͜ʒoŋ", "do", ""),
    (76, "turtle", "4", "katʰua̯", "km", ""),
    (76, "turtle", "4", "kʰatʼtʰua̯", "efhn", ""),
    (76, "turtle", "4", "kʰatua̯", "g", ""),
    (76, "turtle", "5", "haru", "p", ""),
    (76, "turtle", "6", "dkar", "jk", ""),
    (76, "turtle", "7", "kʰuʃum", "l", ""),
    (76, "turtle", "8", "kɔttʃʰop", "0", ""),

    (77, "frog", "1", "luklak", "lm", ""),
    (77, "frog", "1", "luwakʼ", "bc", ""),
    (77, "frog", "2", "bɛŋboŋ", "deikmn", ""),
    (77, "frog", "3", "heruʔ", "p", ""),
    (77, "frog", "4", "bɛŋ", "0afghjo", ""),

    (78, "dog", "1", "atʃak", "adefghino", ""),
    (78, "dog", "2", "kɨi", "lm", ""),
    (78, "dog", "2", "kui", "bc", ""),
    (78, "dog", "3", "kʰsu", "p", ""),
    (78, "dog", "3", "ksu", "jk", ""),
    (78, "dog", "4", "kukur", "0", ""),

    (79, "cat", "1", "mɛŋgao̯", "fjk", ""),
    (79, "cat", "1", "mɛŋgo", "gh", ""),
    (79, "cat", "1", "mɛŋgoŋ", "aeio", ""),
    (79, "cat", "1", "mɛŋgou̯", "dn", ""),
    (79, "cat", "2", "bɨi̯ra", "lm", ""),
    (79, "cat", "2", "bilai̯", "bc", ""),
    (79, "cat", "2", "biral", "0", ""),
    (79, "cat", "3", "mio̯", "p", ""),

    (80, "cow", "1", "mɨʔsɨu̯", "k", ""),
    (80, "cow", "1", "maʔʃu", "aefgn", ""),
    (80, "cow", "1", "maʔsɨu̯", "j", ""),
    (80, "cow", "1", "maʔsu", "bcdhilm", ""),
    (80, "cow", "1", "maʔtʃʰu", "o", ""),
    (80, "cow", "2", "goru", "0", ""),
    (80, "cow", "3", "mɨʔsɨu̯", "k", ""),
    (80, "cow", "3", "mɨsɨ", "p", ""),
    (80, "cow", "3", "maʔsɨu̯", "j", ""),
    (80, "cow", "3", "maʔsu", "bcdhilm", ""),
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

    expected = {(item, code) for item in range(76, 81) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(76, 81):
        gloss, pdf_page, printed_page, column = metadata[item]
        for code in SITE_CODES:
            source_lines = by_cell[(item, code)]
            attested = [row for row in source_lines if row["Line_Status"] == "attested"]
            blanks = [row for row in source_lines if row["Line_Status"] == "source_blank"]
            assert not (attested and blanks)
            assert attested
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
    assert len(lines) == 40
    assert len(cells) == 85
    assert all(row["Review_Status"] == "attested" for row in cells)
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells) == 97
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
