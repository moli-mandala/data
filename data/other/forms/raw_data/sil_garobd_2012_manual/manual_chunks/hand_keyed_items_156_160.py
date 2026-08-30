#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 156-160.

Every response, group number, bracket code, repetition, and the whole-item
"[not used]" disposition below was independently hand-keyed from physical PDF
pages 72-73 / printed pages 65-66. The 300-dpi pages were primary and small IPA
marks were checked in targeted 1200-dpi crops. OCR, PDF text, legacy data,
installed forms, and earlier audits are not inputs to this script.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
LINES_OUTPUT = HERE / "items_156_160_lines.tsv"
CELLS_OUTPUT = HERE / "items_156_160_cells.tsv"
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
    156: ("72", "65", "right"),
    **{item: ("73", "66", "left") for item in range(157, 161)},
}

# One tuple per printed response or whole-item disposition line, in source
# order. The all-site code string on item 159 is a mechanical representation
# of the source's whole-item scope, not a printed bracket code.
PRINTED_LINES = [
    (156, "thread", "1", "kʰɨldɨŋ", "aio", ""),
    (156, "thread", "1", "kʰuldɨŋ", "defn", ""),
    (156, "thread", "1", "kʰuntɨŋ", "bc", ""),
    (156, "thread", "2", "kʰɨl", "h", ""),
    (156, "thread", "2", "kʰul", "g", ""),
    (156, "thread", "3", "kʃai̯", "jk", ""),
    (156, "thread", "4", "pitɨŋ", "lm", ""),
    (156, "thread", "5", "ʃuta", "0", ""),
    (156, "thread", "6", "kʰsai̯", "p", ""),

    (157, "broom", "1", "ʃaʔla", "aio", ""),
    (157, "broom", "1", "ʃaʔlakʼ", "ef", ""),
    (157, "broom", "1", "satʼla", "d", ""),
    (157, "broom", "2", "nohɛkʼ", "bc", ""),
    (157, "broom", "3", "d͜ʒatʼta", "gh", ""),
    (157, "broom", "4", "tʃipʼnatʼ", "jk", ""),
    (157, "broom", "4", "tʃipnat", "p", ""),
    (157, "broom", "5", "nogɛkʼ", "lm", ""),
    (157, "broom", "6", "ʃatʃuni", "n", ""),
    (157, "broom", "7", "d͜ʒʰaɾu", "0", ""),

    (158, "spoon (for eating)", "1", "kortʃali", "b", ""),
    (158, "spoon (for eating)", "1", "kortʃila", "c", ""),
    (158, "spoon (for eating)", "2", "ata", "dn", ""),
    (158, "spoon (for eating)", "3", "tʃamotʃ", "0aefghijklmnop", ""),

    (159, "knife (to cut meat)", "", "", "0abcdefghijklmnop", "[not used]"),

    (160, "hammer", "1", "hatur", "abcdeghlmno", ""),
    (160, "hammer", "1", "haturi", "0ij", ""),
    (160, "hammer", "2", "d͜ʒoŋ mnoʔ", "k", ""),
    (160, "hammer", "3", "tɨrnim", "p", ""),
    (160, "hammer", "4", "atur", "f", ""),
    (160, "hammer", "4", "hatur", "abcdeghlmno", ""),
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
        status = "not_used" if status_text == "[not used]" else "attested"
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

    expected = {(item, code) for item in range(156, 161) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(156, 161):
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
    assert len(lines) == 30
    assert len(cells) == 85
    assert sum(row["Review_Status"] == "attested" for row in cells) == 68
    assert sum(row["Review_Status"] == "not_used" for row in cells) == 17
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 80
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed response/disposition lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
