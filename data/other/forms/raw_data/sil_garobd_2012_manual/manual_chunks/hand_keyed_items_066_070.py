#!/usr/bin/env python3
"""Freeze the OCR-blind Garo ESR 2012-007 ledgers for items 66-70.

Every response, group number, bracket code, and explicit blank below was
independently hand-keyed from physical PDF page 60 / printed page 53. The
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
LINES_OUTPUT = HERE / "items_066_070_lines.tsv"
CELLS_OUTPUT = HERE / "items_066_070_cells.tsv"
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
    66: ("60", "53", "left"),
    **{item: ("60", "53", "right") for item in range(67, 71)},
}

# One tuple per printed response line, in source order.
PRINTED_LINES = [
    (66, "pepper", "1", "d͜ʒɨrɨk", "lm", ""),
    (66, "pepper", "1", "d͜ʒaʔlukʼ", "fg", ""),
    (66, "pepper", "1", "d͜ʒalɨkʼ", "ao", ""),
    (66, "pepper", "2", "d͜ʒaʔalɨkʼ", "h", ""),
    (66, "pepper", "2", "d͜ʒaʔlukʼ", "fg", ""),
    (66, "pepper", "2", "d͜ʒaʔlukʰa", "e", ""),
    (66, "pepper", "2", "d͜ʒalɨkʼ", "ao", ""),
    (66, "pepper", "2", "d͜ʒalɨkʰa", "i", ""),
    (66, "pepper", "2", "tʃaʔlukʼ", "dn", ""),
    (66, "pepper", "3", "morɨtʃ", "0", ""),
    (66, "pepper", "4", "murtʃukʼ", "bc", ""),
    (66, "pepper", "5", "soʔʃat", "jk", ""),
    (66, "pepper", "5", "suʔ sat sao̯", "p", ""),

    (67, "elephant", "0", "", "p", "no entry"),
    (67, "elephant", "1", "moŋma", "adilm", ""),
    (67, "elephant", "1", "moŋmau̯", "o", ""),
    (67, "elephant", "2", "jao̯ba", "jk", ""),
    (67, "elephant", "3", "hatɨ", "0bcefghn", ""),

    (68, "tiger", "1", "matʼsa", "bce", ""),
    (68, "tiger", "1", "matʼtʃʰa", "adfghilmno", ""),
    (68, "tiger", "2", "kʰla", "jkp", ""),
    (68, "tiger", "3", "bagʰ", "0", ""),

    (69, "bear", "1", "makʼbɨl", "io", ""),
    (69, "bear", "1", "makʼbul", "deflmn", ""),
    (69, "bear", "1", "makʼpɨl", "agh", ""),
    (69, "bear", "2", "dɨŋ ŋem", "p", ""),
    (69, "bear", "2", "dɨŋom", "jk", ""),
    (69, "bear", "3", "baluk", "bc", ""),
    (69, "bear", "3", "bʰaluk", "0", ""),

    (70, "deer", "0", "", "a", "no entry"),
    (70, "deer", "1", "matʃao̯", "bc", ""),
    (70, "deer", "2", "matʼtʃʰok", "io", ""),
    (70, "deer", "2", "matʼtʃok", "defghlmn", ""),
    (70, "deer", "3", "sor", "jk", ""),
    (70, "deer", "4", "horɨn", "0", ""),
    (70, "deer", "5", "skao̯", "p", ""),
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

    expected = {(item, code) for item in range(66, 71) for code in SITE_CODES}
    assert set(by_cell) == expected

    out: list[dict[str, str]] = []
    for item in range(66, 71):
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
    assert len(lines) == 36
    assert len(cells) == 85
    assert sum(row["Review_Status"] == "attested" for row in cells) == 83
    assert sum(row["Review_Status"] == "source_blank" for row in cells) == 2
    assert sum(len(row["Manual_Transcription"].split(" | ")) for row in cells if row["Manual_Transcription"]) == 87
    write_tsv(LINES_OUTPUT, LINE_FIELDS, lines)
    write_tsv(CELLS_OUTPUT, CELL_FIELDS, cells)
    print(f"wrote {len(lines)} printed lines to {LINES_OUTPUT}")
    print(f"wrote {len(cells)} expanded cells to {CELLS_OUTPUT}")


if __name__ == "__main__":
    main()
