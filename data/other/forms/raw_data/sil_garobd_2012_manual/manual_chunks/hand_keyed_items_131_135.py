#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 131-135.

Every response, group number, bracket code, and repetition below was
independently hand-keyed from physical PDF pages 69-70 / printed pages 62-63.
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
LINES_OUTPUT = HERE / "items_131_135_lines.tsv"
CELLS_OUTPUT = HERE / "items_131_135_cells.tsv"
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
    **{item: ("69", "62", "right") for item in range(131, 133)},
    **{item: ("70", "63", "left") for item in range(133, 136)},
}

# One tuple per printed response line, in source order.
PRINTED_LINES = [
    (131, "mother", "1", "ama", "adefilmo", ""),
    (131, "mother", "1", "amai̯", "bc", ""),
    (131, "mother", "2", "bɨi̯", "j", ""),
    (131, "mother", "2", "bei̯", "p", ""),
    (131, "mother", "3", "ma", "0", ""),
    (131, "mother", "3", "mao̯", "k", ""),
    (131, "mother", "4", "bai̯", "ghn", ""),
    (131, "mother", "4", "bei̯", "p", ""),

    (132, "husband", "1", "ʃai̯", "ef", ""),
    (132, "husband", "1", "ʃɛ", "adio", ""),
    (132, "husband", "1", "ʃei̯", "gn", ""),
    (132, "husband", "2", "mitʰala", "bc", ""),
    (132, "husband", "3", "koraŋ", "jkp", ""),
    (132, "husband", "4", "d͜ʒɨkʼbipʰa", "lm", ""),
    (132, "husband", "5", "sɨi̯", "h", ""),
    (132, "husband", "6", "ʃami", "0", ""),

    (133, "wife", "1", "d͜ʒɨkʼ", "aghio", ""),
    (133, "wife", "1", "d͜ʒikʼ", "defn", ""),
    (133, "wife", "2", "mitʃikʼ", "bc", ""),
    (133, "wife", "3", "kontʰao̯", "jkp", ""),
    (133, "wife", "4", "d͜ʒɨkʼgɨwuɨ̯", "lm", ""),
    (133, "wife", "5", "stri", "0", ""),

    (134, "son", "1", "mɛʔaʃa biʃa", "o", ""),
    (134, "son", "1", "mɛʔaʃa piʃa", "i", ""),
    (134, "son", "1", "mɛʔɛʃa pʰiʃa", "a", ""),
    (134, "son", "2", "mɛʔaʃa dei̯", "n", ""),
    (134, "son", "2", "mɛʔaʃa doi̯", "e", ""),
    (134, "son", "2", "mɛʔasa dei̯", "h", ""),
    (134, "son", "2", "mɛʔɛʃa dɛ", "d", ""),
    (134, "son", "2", "miʔa doi̯", "f", ""),
    (134, "son", "2", "miʔaʃa dei̯", "g", ""),
    (134, "son", "3", "piʃa", "bc", ""),
    (134, "son", "4", "kʰon koraŋ", "jkp", ""),
    (134, "son", "5", "ʃa bipʰa", "lm", ""),
    (134, "son", "6", "tʃʰɛlɛ", "0", ""),

    (135, "daughter", "1", "mid͜ʒɨk", "a", ""),
    (135, "daughter", "2", "(tiri) piʃa", "c", ""),
    (135, "daughter", "2", "piʃa (tiri)", "b", ""),
    (135, "daughter", "3", "mitʃɨkʼʃa dɛ", "d", ""),
    (135, "daughter", "3", "mitʃɨkʼsa dei̯", "gh", ""),
    (135, "daughter", "3", "mitʃʰikʼ doi̯", "f", ""),
    (135, "daughter", "3", "mitʃʰikʼʃa dei̯", "e", ""),
    (135, "daughter", "3", "mitʃikʼʃa dei̯", "n", ""),
    (135, "daughter", "4", "mitʃɨkʼbiʃa", "o", ""),
    (135, "daughter", "4", "mitʃɨkʼpiʃa", "i", ""),
    (135, "daughter", "5", "kʰon rokmao̯", "jk", ""),
    (135, "daughter", "6", "ʃa gɨwuɨ̯", "lm", ""),
    (135, "daughter", "7", "mɛjɛ", "0", ""),
    (135, "daughter", "8", "rao̯k mao̯", "p", ""),
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

    expected = {(item, code) for item in range(131, 136) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(131, 136):
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
    assert len(lines) == 49
    assert len(cells) == 85
    assert all(row["Review_Status"] == "attested" for row in cells)
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells) == 86
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
