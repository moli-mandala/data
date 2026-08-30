#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 86-90.

Every response, group number, bracket code, repetition, and explicit blank below
was independently hand-keyed from physical PDF page 63 / printed page 56. The
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
LINES_OUTPUT = HERE / "items_086_090_lines.tsv"
CELLS_OUTPUT = HERE / "items_086_090_cells.tsv"
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
    **{item: ("63", "56", "left") for item in range(86, 89)},
    **{item: ("63", "56", "right") for item in range(89, 91)},
}

# One tuple per printed response line, in source order.
PRINTED_LINES = [
    (86, "rat", "1", "miʃi", "adio", ""),
    (86, "rat", "1", "moʃai̯", "bcef", ""),
    (86, "rat", "1", "moʃei̯", "n", ""),
    (86, "rat", "1", "mosei̯", "gh", ""),
    (86, "rat", "2", "kʰnai̯", "jkp", ""),
    (86, "rat", "3", "miʔtʃutʼ", "m", ""),
    (86, "rat", "3", "mid͜ʒutʼ", "l", ""),
    (86, "rat", "4", "idur", "0", ""),

    (87, "chicken (adult female)", "1", "do", "dio", ""),
    (87, "chicken (adult female)", "1", "dou̯", "ghn", ""),
    (87, "chicken (adult female)", "2", "sɨʔer", "p", ""),
    (87, "chicken (adult female)", "2", "siʔɛr", "k", ""),
    (87, "chicken (adult female)", "2", "siɛr", "j", ""),
    (87, "chicken (adult female)", "3", "dau̯", "ef", ""),
    (87, "chicken (adult female)", "4", "tau̯", "bclm", ""),
    (87, "chicken (adult female)", "5", "duʔu", "a", ""),
    (87, "chicken (adult female)", "6", "murgi", "0", ""),

    (88, "egg", "0", "", "p", "no entry"),
    (88, "egg", "1", "bɨtʼtʃi", "aghio", ""),
    (88, "egg", "1", "biʔtʃʰi", "e", ""),
    (88, "egg", "2", "pitɪk", "bc", ""),
    (88, "egg", "3", "dao̯tʃʰi", "f", ""),
    (88, "egg", "3", "dotʃʰi", "dn", ""),
    (88, "egg", "4", "pliŋ", "jk", ""),
    (88, "egg", "5", "tɨi̯", "lm", ""),
    (88, "egg", "6", "d̪im", "0", ""),

    (89, "fish", "1", "naʔtʰokʼ", "adefio", ""),
    (89, "fish", "2", "na", "bcghlmn", ""),
    (89, "fish", "3", "kʰa", "jkp", ""),
    (89, "fish", "4", "matʃʰ", "0", ""),

    (90, "duck", "1", "gagakʼ", "aefgio", ""),
    (90, "duck", "2", "baŋsu", "bc", ""),
    (90, "duck", "3", "doʔɛtʼ", "h", ""),
    (90, "duck", "3", "doʔwatʼ", "dn", ""),
    (90, "duck", "4", "dao̯gɛpʼ", "jkm", ""),
    (90, "duck", "4", "dao̯gɛtʼ", "l", ""),
    (90, "duck", "4", "dao̯gep", "p", ""),
    (90, "duck", "4", "dogɛpʼ", "o", ""),
    (90, "duck", "5", "haʃ", "0", ""),
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

    expected = {(item, code) for item in range(86, 91) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(86, 91):
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
    assert len(lines) == 39
    assert len(cells) == 85
    assert sum(row["Review_Status"] == "attested" for row in cells) == 84
    assert sum(row["Review_Status"] == "source_blank" for row in cells) == 1
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 85
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
