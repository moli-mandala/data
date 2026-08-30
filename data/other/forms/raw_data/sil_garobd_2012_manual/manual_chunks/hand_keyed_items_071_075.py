#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 71-75.

Every response, group number, bracket code, repetition, and explicit blank below
was independently hand-keyed from physical PDF pages 60-61 / printed pages
53-54. The 300-dpi pages were primary and small IPA marks were checked in
targeted 1200-dpi crops. OCR, PDF text, legacy data, installed forms, and earlier
audits are not inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_071_075_lines.tsv"
CELLS_OUTPUT = HERE / "items_071_075_cells.tsv"
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
    71: ("60", "53", "right"),
    **{item: ("61", "54", "left") for item in range(72, 74)},
    **{item: ("61", "54", "right") for item in range(74, 76)},
}

# One tuple per printed response line, in source order.
PRINTED_LINES = [
    (71, "monkey", "0", "", "f", "no entry"),
    (71, "monkey", "1", "hamakʼ", "ailmo", ""),
    (71, "monkey", "2", "kao̯ i", "bc", ""),
    (71, "monkey", "3", "bandor", "dehn", ""),
    (71, "monkey", "3", "banor", "0g", ""),
    (71, "monkey", "4", "tʃɨ riʔ", "p", ""),
    (71, "monkey", "4", "tʃrɨʔ", "jk", ""),

    (72, "rabbit", "0", "", "lop", "no entry"),
    (72, "rabbit", "1", "kʰorgoʃ", "0fgm", ""),
    (72, "rabbit", "1", "kʰorgoʃ", "adj", ""),
    (72, "rabbit", "1", "kʰurguʃ", "en", ""),
    (72, "rabbit", "2", "hed͜ʒabari", "bc", ""),
    (72, "rabbit", "3", "nɛtroŋ", "h", ""),
    (72, "rabbit", "4", "pʰutʼtʰa", "i", ""),
    (72, "rabbit", "5", "tʃahurɨn", "jk", ""),

    (73, "snake", "1", "tʃɨpʼbu", "h", ""),
    (73, "snake", "1", "tʃɨpʼpʰu", "di", ""),
    (73, "snake", "1", "tʃɨpʼpu", "a", ""),
    (73, "snake", "1", "tʃʰɨpʼpʰu", "io", ""),
    (73, "snake", "1", "tʃupʼbu", "fn", ""),
    (73, "snake", "1", "tʃupʼpu", "g", ""),
    (73, "snake", "2", "dɨpɨu̯", "lm", ""),
    (73, "snake", "2", "duɸu", "c", ""),
    (73, "snake", "2", "dupʰu", "b", ""),
    (73, "snake", "3", "tʃɨpʼbu", "h", ""),
    (73, "snake", "3", "tʃɨpʼpʰu", "di", ""),
    (73, "snake", "3", "tʃɨpʼpu", "a", ""),
    (73, "snake", "3", "tʃubu", "e", ""),
    (73, "snake", "3", "tʃupʼbu", "fn", ""),
    (73, "snake", "3", "tʃupʼpu", "g", ""),
    (73, "snake", "4", "bsɨi̯n", "jk", ""),
    (73, "snake", "4", "msei̯n", "p", ""),
    (73, "snake", "5", "ʃap", "0", ""),
    (73, "snake", "6", "dɨpɨu̯", "lm", ""),
    (73, "snake", "6", "dupʰu", "b", ""),
    (73, "snake", "6", "tʃubu", "e", ""),

    (74, "crocodile", "1", "arɨŋga", "i", ""),
    (74, "crocodile", "1", "arɨŋkʰa", "a", ""),
    (74, "crocodile", "1", "arɛŋga", "efno", ""),
    (74, "crocodile", "1", "arɛŋkʰa", "d", ""),
    (74, "crocodile", "1", "arɪŋga", "jkm", ""),
    (74, "crocodile", "1", "harɪŋga", "l", ""),
    (74, "crocodile", "1", "hariŋga", "p", ""),
    (74, "crocodile", "2", "kumir", "0bcgh", ""),

    (75, "house lizard", "0", "", "lop", "no entry"),
    (75, "house lizard", "1", "kʰantʃʰidɨk", "ai", ""),
    (75, "house lizard", "2", "nok dabrɛk", "bc", ""),
    (75, "house lizard", "3", "malɛŋkʰao̯", "jk", ""),
    (75, "house lizard", "4", "antɨka", "g", ""),
    (75, "house lizard", "4", "antɛkʼka", "e", ""),
    (75, "house lizard", "4", "antika", "h", ""),
    (75, "house lizard", "4", "hantɨkʼka", "d", ""),
    (75, "house lizard", "4", "hantɪkʼka", "n", ""),
    (75, "house lizard", "4", "hantika", "f", ""),
    (75, "house lizard", "5", "toktokkorot", "m", ""),
    (75, "house lizard", "6", "tɪktɪkki", "0", ""),
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

    expected = {(item, code) for item in range(71, 76) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(71, 76):
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
    assert len(lines) == 56
    assert len(cells) == 85
    assert sum(row["Review_Status"] == "attested" for row in cells) == 78
    assert sum(row["Review_Status"] == "source_blank" for row in cells) == 7
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 91
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
