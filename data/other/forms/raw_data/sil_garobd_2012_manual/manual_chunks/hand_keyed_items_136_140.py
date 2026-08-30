#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 136-140.

Every response, group number, bracket code, and repetition below was
independently hand-keyed from physical PDF pages 70-71 / printed pages 63-64.
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
LINES_OUTPUT = HERE / "items_136_140_lines.tsv"
CELLS_OUTPUT = HERE / "items_136_140_cells.tsv"
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
    **{item: ("70", "63", "right") for item in range(136, 140)},
    140: ("71", "64", "left"),
}

# One tuple per printed response line, in source order. These values are the
# frozen manual reading; bracket codes are expanded mechanically below.
PRINTED_LINES = [
    (136, "elder brother", "1", "ada", "fio", ""),
    (136, "elder brother", "1", "dada", "acdeghln", ""),
    (136, "elder brother", "2", "kaka", "bc", ""),
    (136, "elder brother", "3", "hmɨn", "k", ""),
    (136, "elder brother", "3", "hmɨn (kɨnba)", "j", ""),
    (136, "elder brother", "4", "pʰao̯ tʃuŋguwa", "m", ""),
    (136, "elder brother", "5", "bɔro bʰai̯", "0", ""),
    (136, "elder brother", "6", "hɨn min koraŋ", "p", ""),

    (137, "elder sister", "1", "abi", "adio", ""),
    (137, "elder sister", "2", "ad͜ʒa", "bcl", ""),
    (137, "elder sister", "3", "bai̯", "cef", ""),
    (137, "elder sister", "4", "bortʰao̯ kɨnba", "jk", ""),
    (137, "elder sister", "5", "d͜ʒa tʃuŋguwa", "m", ""),
    (137, "elder sister", "6", "bɔro bon / didi", "0gn", ""),
    (137, "elder sister", "6", "dadi", "h", ""),
    (137, "elder sister", "7", "hɨn min rao̯k mao̯", "p", ""),

    (138, "younger brother", "1", "d͜ʒoŋ", "abcefgo", ""),
    (138, "younger brother", "2", "d͜ʒoŋgoa̯", "l", ""),
    (138, "younger brother", "3", "nono", "h", ""),
    (138, "younger brother", "4", "hɨmbu dodɨpʼ", "jk", ""),
    (138, "younger brother", "4", "hɨnbu dudit", "p", ""),
    (138, "younger brother", "5", "d͜ʒoŋ mɨlguwa", "m", ""),
    (138, "younger brother", "6", "tʃʰoto bʰai̯", "0", ""),
    (138, "younger brother", "7", "d͜ʒod͜ʒoŋ", "den", ""),
    (138, "younger brother", "8", "and͜ʒoŋ", "i", ""),

    (139, "younger sister", "1", "anu", "ai", ""),
    (139, "younger sister", "1", "nono", "deghlno", ""),
    (139, "younger sister", "2", "nau̯", "bcf", ""),
    (139, "younger sister", "3", "bortʰao̯ dodɨpʼ", "jk", ""),
    (139, "younger sister", "4", "nao̯ mɨlguwa", "m", ""),
    (139, "younger sister", "5", "tʃʰoto bon", "0", ""),
    (139, "younger sister", "6", "hɨnbu rao̯kmao̯", "p", ""),

    (140, "friend", "1", "bad͜ʒu", "aefghiklmn", ""),
    (140, "friend", "2", "ʃaŋgra", "b", ""),
    (140, "friend", "2", "saŋgra", "c", ""),
    (140, "friend", "3", "bɛʃa", "d", ""),
    (140, "friend", "3", "bei̯ʃa", "g", ""),
    (140, "friend", "4", "ma lɔk", "p", ""),
    (140, "friend", "4", "marlokʼ", "j", ""),
    (140, "friend", "5", "rɨpɛŋ", "o", ""),
    (140, "friend", "6", "bondʰu", "0", ""),
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

    expected = {(item, code) for item in range(136, 141) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(136, 141):
        gloss, pdf_page, printed_page, column = metadata[item]
        for code in SITE_CODES:
            source_lines = by_cell[(item, code)]
            attested = [row for row in source_lines if row["Line_Status"] == "attested"]
            blanks = [row for row in source_lines if row["Line_Status"] == "source_blank"]
            assert not blanks
            row = {
                "Item": str(item), "Gloss": gloss, "Site_Code": code,
                "Site_Identity": f"printed site code {code}", "PDF_Page": pdf_page,
                "Printed_Page": printed_page, "Column": column,
                "Manual_Transcription": " | ".join(line["Manual_Transcription"] for line in attested),
                "Similarity_Groups": "|".join(line["Similarity_Group"] for line in attested),
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
    assert len(lines) == 41
    assert len(cells) == 85
    assert all(row["Review_Status"] == "attested" for row in cells)
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells) == 89
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
