"""Extract the Unicode wordlists in Hilty & Mitchell's Yamphu survey.

Appendix C.4 of the born-digital SIL PDF (PDF pages 116--136, printed
pages 109--129) contains a 252-item comparative list for nine sites.  The
table is rotated, so this parser works from the positioned Unicode text
lines rather than OCR.

Download ``silesr2014_007.pdf`` to ``tmp/pdfs/yamphu`` before running.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "yamphu" / "silesr2014_007.pdf"
OUTPUT = HERE.parent / "20260813-yamphu.csv"
SOURCE = "hilty-mitchell2014"

LECTS = {
    "LA": "lohorung_angala",
    "LD": "lohorung_dhupu",
    "GP": "lohorung_gairi_pangma",
    "YD": "southern_yamphu_devitar",
    "YR": "southern_yamphu_rajarani",
    "YH": "yamphu_hedangna",
    "YK": "yamphu_khoktak",
    "YN": "yamphu_num",
    "YS": "yamphu_seduwa",
}

COLUMN_BANDS = ((60, 205), (200, 344), (340, 480))
CODE_RE = re.compile(r"[A-Z/]+$")


def _vertical_lines(page: fitz.Page) -> list[dict]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            if tuple(round(value, 2) for value in line["dir"]) != (0.0, -1.0):
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                lines.append({"text": text, "x": line["bbox"][0], "y": line["bbox"][3]})
    return lines


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int]]:
    """Return (concept number, gloss, site code, form, printed page)."""
    document = fitz.open(pdf_path)
    records: list[tuple[int, str, str, str, int]] = []
    concept_number = 0

    # Zero-based 115:136 = PDF pages 116--136 = printed pages 109--129.
    for page_index in range(115, 136):
        lines = _vertical_lines(document[page_index])
        for column_index, (left, right) in enumerate(COLUMN_BANDS):
            column = [line for line in lines if left <= line["x"] < right]
            row_y = sorted(
                {
                    round(line["y"], 1)
                    for line in column
                    if "LA" in line["text"] and CODE_RE.fullmatch(line["text"])
                },
                reverse=True,
            )
            if len(row_y) != 4:
                raise ValueError(
                    f"Expected four wordlist rows in column {column_index + 1} "
                    f"on PDF page {page_index + 1}, got {len(row_y)}"
                )

            for row_index, top_y in enumerate(row_y):
                bottom_y = row_y[row_index + 1] if row_index + 1 < 4 else 20.0
                row = [
                    line
                    for line in column
                    if bottom_y + 0.2 < line["y"] <= top_y + 0.2
                ]
                labels = sorted(
                    (
                        line
                        for line in row
                        if abs(line["y"] - top_y) <= 0.2
                        and CODE_RE.fullmatch(line["text"])
                    ),
                    key=lambda line: line["x"],
                )
                codes = [code for label in labels for code in label["text"].split("/")]
                if sorted(codes) != sorted(LECTS):
                    raise ValueError(
                        f"Bad site-code row on PDF page {page_index + 1}: {codes}"
                    )

                label_x = [label["x"] for label in labels]
                glosses = [
                    line
                    for line in row
                    if all(abs(line["x"] - x) > 2 for x in label_x)
                    and line["text"] != "-"
                    and not re.fullmatch(r"[\d, ]+", line["text"])
                ]
                # The source leaves this one gloss cell blank, but its position
                # in the numbered elicitation list identifies it unambiguously.
                if (page_index, column_index, row_index) == (135, 1, 1):
                    gloss = "you (dual/H)"
                elif len(glosses) == 1:
                    gloss = _clean(glosses[0]["text"])
                else:
                    raise ValueError(
                        f"Expected one gloss on PDF page {page_index + 1}, "
                        f"column {column_index + 1}, row {row_index + 1}: {glosses}"
                    )

                concept_number += 1
                printed_page = page_index - 6
                for label in labels:
                    label_codes = label["text"].split("/")
                    candidates = sorted(
                        (
                            line
                            for line in row
                            if abs(line["x"] - label["x"]) <= 0.5
                            and bottom_y + 0.2 < line["y"] < top_y - 0.2
                        ),
                        key=lambda line: line["y"],
                        reverse=True,
                    )
                    # A combined label (e.g. LA/LD) explicitly marks both sites
                    # as lacking a form and therefore has no transcription line.
                    if len(label_codes) > 1:
                        continue
                    # Some ordinary (uncombined) site labels also have a blank
                    # cell; the printed table uses blanks and hyphens alike.
                    if not candidates:
                        continue
                    form = _clean(candidates[0]["text"])
                    if form == "-":
                        continue
                    for alternate in re.split(r"\s*/\s*", form):
                        if alternate:
                            records.append(
                                (
                                    concept_number,
                                    gloss,
                                    label_codes[0],
                                    alternate,
                                    printed_page,
                                )
                            )

    if concept_number != 252:
        raise ValueError(f"Expected 252 concepts, got {concept_number}")
    if len(records) != 2188:
        raise ValueError(f"Expected 2,188 printed forms, got {len(records)}")
    return records


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, code, form, printed_page in extracted:
        occurrence[(concept, code)] += 1
        rows.append(
            [
                LECTS[code],
                "",
                form,
                gloss,
                "",
                form,
                "",
                f"{SOURCE}[p. {printed_page}]",
                "",
                "",
                f"yamphu:{concept}:{code.lower()}:{occurrence[(concept, code)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
