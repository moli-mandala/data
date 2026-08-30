#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 16-20.

Every response, similarity-group number, bracket code, and explicit blank below
was independently hand-keyed from physical PDF page 54 / printed page 47. The
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
LINES_OUTPUT = HERE / "items_016_020_lines.tsv"
CELLS_OUTPUT = HERE / "items_016_020_cells.tsv"
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
    16: ("54", "47", "left"),
    17: ("54", "47", "left"),
    18: ("54", "47", "left"),
    19: ("54", "47", "right"),
    20: ("54", "47", "right"),
}

# One tuple per printed response line, in source order:
# (item, gloss, similarity group, diplomatic response, bracket codes, status text).
# Repeated responses and repeated site assignments are deliberately preserved.
PRINTED_LINES = [
    (16, "mud", "1", "haʔapɛkʼ", "adeg", ""),
    (16, "mud", "1", "haʔpɛkʼ", "f", ""),
    (16, "mud", "2", "haʔapɛkʼ", "adeg", ""),
    (16, "mud", "2", "hadɨbɛkʼ", "i", ""),
    (16, "mud", "3", "haʔdilɛka / kadoŋ", "bc", ""),
    (16, "mud", "3", "hadɨbɛkʼ", "i", ""),
    (16, "mud", "4", "doba", "lm", ""),
    (16, "mud", "5", "dir", "jk", ""),
    (16, "mud", "5", "ədir", "p", ""),
    (16, "mud", "6", "haʔdilɛka / kadoŋ", "bc", ""),
    (16, "mud", "6", "kada", "0", ""),
    (16, "mud", "7", "pɛk", "ghno", ""),

    (17, "dust", "0", "", "f", "no entry"),
    (17, "dust", "1", "dɨmut", "p", ""),
    (17, "dust", "2", "hapu", "b", ""),
    (17, "dust", "2", "haputʼ", "c", ""),
    (17, "dust", "3", "hantʃiŋ", "d", ""),
    (17, "dust", "4", "haʔadula", "e", ""),
    (17, "dust", "4", "hadula", "m", ""),
    (17, "dust", "4", "hagundula", "a", ""),
    (17, "dust", "5", "hantʃʰɛŋ", "i", ""),
    (17, "dust", "6", "mɛŋpɨrpʰu", "jk", ""),
    (17, "dust", "7", "habukʰu", "l", ""),
    (17, "dust", "8", "dʰula", "0ghno", ""),

    (18, "stone", "1", "loŋtʰai̯", "bc", ""),
    (18, "stone", "1", "roŋtʰai̯", "eflm", ""),
    (18, "stone", "1", "roŋtʰi", "adghino", ""),
    (18, "stone", "2", "mao̯", "jkp", ""),
    (18, "stone", "3", "patʰor", "0", ""),

    (19, "sand", "1", "haŋtʃʰɛŋ", "ao", ""),
    (19, "sand", "1", "haŋʔtʃɛŋ", "efgm", ""),
    (19, "sand", "1", "haŋtʃɛŋ", "n", ""),
    (19, "sand", "1", "haŋtʃʰɛŋ", "i", ""),
    (19, "sand", "1", "hantʃiŋ", "d", ""),
    (19, "sand", "1", "hatʃɛŋ", "cl", ""),
    (19, "sand", "2", "haŋtʃʰɛŋ", "ao", ""),
    (19, "sand", "2", "haŋtʃɛŋ", "n", ""),
    (19, "sand", "2", "haŋtʃʰɛŋ", "i", ""),
    (19, "sand", "2", "hantʃiŋ", "d", ""),
    (19, "sand", "2", "hasɛŋ", "b", ""),
    (19, "sand", "2", "hatʃɛŋ", "cl", ""),
    (19, "sand", "3", "dulabali", "h", ""),
    (19, "sand", "4", "ɖʒia̯p", "j", ""),
    (19, "sand", "4", "ɖʒiɛp", "k", ""),
    (19, "sand", "5", "bali", "0", ""),
    (19, "sand", "6", "ɖʒmia̯k", "p", ""),

    (20, "gold", "1", "kʰsiar", "p", ""),
    (20, "gold", "1", "ksɛr", "jk", ""),
    (20, "gold", "2", "ʃona", "0abcdefghilmno", ""),
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

    expected = {(item, code) for item in range(16, 21) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(16, 21):
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
                "Uncertainty": "",
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
    assert len(lines) == 49
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
