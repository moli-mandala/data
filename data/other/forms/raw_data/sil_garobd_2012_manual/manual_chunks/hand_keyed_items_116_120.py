#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 116-120.

Every response, group number, bracket code, repetition, and explicit blank below
was independently hand-keyed from physical PDF pages 67-68 / printed pages
60-61. The 300-dpi pages were primary and small IPA marks were checked in
targeted 1200-dpi crops. OCR, PDF text, legacy data, installed forms, and earlier
audits are not inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_116_120_lines.tsv"
CELLS_OUTPUT = HERE / "items_116_120_cells.tsv"
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
    **{item: ("67", "60", "right") for item in range(116, 119)},
    **{item: ("68", "61", "left") for item in range(119, 121)},
}

# One tuple per printed response line, in source order.
PRINTED_LINES = [
    (116, "knee", "1", "d͜ʒaʃukʰu", "ag", ""),
    (116, "knee", "1", "d͜ʒaʃuku", "dn", ""),
    (116, "knee", "1", "d͜ʒaskʰu", "o", ""),
    (116, "knee", "2", "d͜ʒaʔʃukʰu", "efhi", ""),
    (116, "knee", "2", "d͜ʒaʃukʰu", "ag", ""),
    (116, "knee", "2", "d͜ʒaʃuku", "dn", ""),
    (116, "knee", "3", "tʃakɾɛŋgila", "bc", ""),
    (116, "knee", "4", "eŋmao̯ɾɛŋ", "jk", ""),
    (116, "knee", "4", "eŋmao̯ria̯ŋ", "p", ""),
    (116, "knee", "5", "tʃakɨu̯", "lm", ""),
    (116, "knee", "6", "hatu", "0", ""),

    (117, "foot", "1", "d͜ʒaʔapʰa", "adghi", ""),
    (117, "foot", "1", "d͜ʒaʔpʰa", "o", ""),
    (117, "foot", "1", "tʃaʔpʰa", "l", ""),
    (117, "foot", "2", "tʃakɾɛŋ aprakʼ", "b", ""),
    (117, "foot", "3", "tʃakɾɛŋ", "c", ""),
    (117, "foot", "4", "tʃaʔtʃokʼ", "e", ""),
    (117, "foot", "5", "d͜ʒapʰatʰai̯", "f", ""),
    (117, "foot", "6", "kid͜ʒia̯t", "p", ""),
    (117, "foot", "6", "kʰd͜ʒɛtʼ", "jk", ""),
    (117, "foot", "7", "tʃaʔ", "m", ""),
    (117, "foot", "8", "d͜ʒaʔtʰɪŋ", "n", ""),
    (117, "foot", "9", "pɔd", "0", ""),

    (118, "bone", "1", "gɛɾɛŋ", "en", ""),
    (118, "bone", "1", "gɾeŋ", "adfghio", ""),
    (118, "bone", "1", "kɛɾɛŋ", "bclm", ""),
    (118, "bone", "2", "tʃɛʔɛŋ", "k", ""),
    (118, "bone", "3", "tʃia̯ŋ", "p", ""),
    (118, "bone", "3", "tʃiɛ̯ŋ", "j", ""),
    (118, "bone", "4", "haɾ", "0", ""),

    (119, "fat", "0", "", "abil", "no entry"),
    (119, "fat", "1", "hokʼpɛkɛn", "c", ""),
    (119, "fat", "2", "mɨtʼdɨm", "dho", ""),
    (119, "fat", "2", "mudum", "ef", ""),
    (119, "fat", "3", "tɛl", "gn", ""),
    (119, "fat", "4", "kʰlɨi̯ŋ", "jk", ""),
    (119, "fat", "4", "kʰlɪŋ", "p", ""),
    (119, "fat", "5", "ludi", "m", ""),
    (119, "fat", "6", "tʃorbi", "0", ""),

    (120, "skin", "1", "bigɨl", "aghio", ""),
    (120, "skin", "1", "bigul", "defn", ""),
    (120, "skin", "2", "kʰopakʼ", "bc", ""),
    (120, "skin", "3", "snɪʔ", "jkp", ""),
    (120, "skin", "4", "kʰol", "lm", ""),
    (120, "skin", "5", "tʃamɾa", "0", ""),
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

    expected = {(item, code) for item in range(116, 121) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(116, 121):
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
    assert len(lines) == 45
    assert len(cells) == 85
    assert sum(row["Review_Status"] == "attested" for row in cells) == 81
    assert sum(row["Review_Status"] == "source_blank" for row in cells) == 4
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 85
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
