#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 11-15.

Every response, similarity-group number, bracket code, and explicit blank below
was independently hand-keyed from physical PDF page 53 / printed page 46. The
300-dpi page was the primary source and small IPA marks were checked in targeted
1200-dpi crops. OCR, PDF text, legacy-font data, installed forms, and earlier
audits are not inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_011_015_lines.tsv"
CELLS_OUTPUT = HERE / "items_011_015_cells.tsv"
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
    11: ("53", "46", "left"),
    12: ("53", "46", "right"),
    13: ("53", "46", "right"),
    14: ("53", "46", "right"),
    15: ("53", "46", "right"),
}

# One tuple per printed response line, in source order:
# (item, gloss, similarity group, diplomatic response, bracket codes, status text).
PRINTED_LINES = [
    (11, "sea", "0", "", "afhn", "no entry"),
    (11, "sea", "1", "ʃaɡal", "do", ""),
    (11, "sea", "1", "ʃaɡor", "0bcegijklm", ""),
    (11, "sea", "2", "duriou̯", "p", ""),

    (12, "mountain", "0", "", "lp", "no entry"),
    (12, "mountain", "1", "brɨŋ", "a", ""),
    (12, "mountain", "2", "bono", "bc", ""),
    (12, "mountain", "3", "haʔroŋɡa", "defn", ""),
    (12, "mountain", "4", "pal", "gh", ""),
    (12, "mountain", "5", "aʔbri", "o", ""),
    (12, "mountain", "5", "haʔbri", "i", ""),
    (12, "mountain", "5", "haʔpri", "m", ""),
    (12, "mountain", "6", "ɨdom", "jk", ""),
    (12, "mountain", "6", "dɔm", "p", ""),
    (12, "mountain", "7", "pahar", "0", ""),

    (13, "water", "1", "tʃʰi", "adefino", ""),
    (13, "water", "2", "tʃi", "gh", ""),
    (13, "water", "3", "tɨi̯", "lm", ""),
    (13, "water", "3", "ti", "bc", ""),
    (13, "water", "4", "ɡum", "jkp", ""),
    (13, "water", "5", "pani", "0", ""),

    (14, "river", "1", "ɡaŋ", "adefghino", ""),
    (14, "river", "2", "dʒora", "b", ""),
    (14, "river", "3", "tʰoloŋ", "c", ""),
    (14, "river", "4", "kmao̯", "jk", ""),
    (14, "river", "5", "tei̯ kʰar", "l", ""),
    (14, "river", "6", "tei̯ muŋ", "m", ""),
    (14, "river", "7", "nodi", "0", ""),
    (14, "river", "8", "pɔr", "p", ""),

    (15, "soil/ground", "1", "haʔ", "bcdefghilm", ""),
    (15, "soil/ground", "1", "haʔa", "ano", ""),
    (15, "soil/ground", "2", "kmɛŋ", "jk", ""),
    (15, "soil/ground", "2", "kmia̯n", "p", ""),
    (15, "soil/ground", "3", "mat̪i", "0", ""),
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
            "Line_ID": f"i{item:03d}-l{order:02d}",
            "Item": str(item),
            "Gloss": gloss,
            "PDF_Page": pdf_page,
            "Printed_Page": printed_page,
            "Column": column,
            "Line_Order": str(order),
            "Similarity_Group": group,
            "Manual_Transcription": form,
            "Bracket_Codes": codes,
            "Printed_Status_Text": status_text,
            "Line_Status": status,
            "Confidence": "high",
            "Uncertainty": "",
            "Reviewer_Method": METHOD,
            "Reviewed_At": "2026-08-29",
            "Reviewer_Declaration": DECLARATION,
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

    expected = {(item, code) for item in range(11, 16) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(11, 16):
        gloss, pdf_page, printed_page, column = metadata[item]
        for code in SITE_CODES:
            source_lines = by_cell[(item, code)]
            attested = [row for row in source_lines if row["Line_Status"] == "attested"]
            blanks = [row for row in source_lines if row["Line_Status"] == "source_blank"]
            if attested and blanks:
                form = " | ".join(row["Manual_Transcription"] for row in attested)
                groups = "|".join(row["Similarity_Group"] for row in attested)
                qualification = 'also printed "no entry" in group 0'
                status = "source_conflict"
                confidence = "high"
                uncertainty = "source prints both group-0 no entry and an attested response"
            elif attested:
                form = " | ".join(row["Manual_Transcription"] for row in attested)
                groups = "|".join(row["Similarity_Group"] for row in attested)
                qualification = ""
                status = "attested"
                confidence = "high"
                uncertainty = ""
            else:
                assert len(blanks) == 1
                form = ""
                groups = blanks[0]["Similarity_Group"]
                qualification = 'printed "no entry"'
                status = "source_blank"
                confidence = "not_applicable"
                uncertainty = ""
            row = {
                "Item": str(item),
                "Gloss": gloss,
                "Site_Code": code,
                "Site_Identity": f"printed site code {code}",
                "PDF_Page": pdf_page,
                "Printed_Page": printed_page,
                "Column": column,
                "Manual_Transcription": form,
                "Similarity_Groups": groups,
                "Source_Line_IDs": "|".join(line["Line_ID"] for line in source_lines),
                "Source_Qualification": qualification,
                "Review_Status": status,
                "Scope": "control_audit_only" if code == "0" else "neutral_unreconciled",
                "Confidence": confidence,
                "Uncertainty": uncertainty,
                "Reviewer_Method": METHOD,
                "Reviewed_At": "2026-08-29",
                "Reviewer_Declaration": DECLARATION,
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
    assert len(lines) == 34
    assert len(cells) == 85
    assert sum(row["Review_Status"] == "attested" for row in cells) == 79
    assert sum(row["Review_Status"] == "source_blank" for row in cells) == 5
    assert sum(row["Review_Status"] == "source_conflict" for row in cells) == 1
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 80
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
