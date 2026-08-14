"""Extract the five target wordlists from Thakur & Thakur (2016).

Annex E of the born-digital survey PDF (PDF pages 124--132, printed
pages 113--121) contains 210 populated prompts for the Sarlahi,
Mahottari, Dhanusha, Saptari, and Morang survey points. The appendix
contains no comparator-language column.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[4] / "tmp" / "pdfs" / "magahi" / "source.pdf"
OUTPUT = HERE.parent / "20260813-magahi.csv"
SOURCE = "thakur-thakur2016magahi"

LECTS = {
    "Sarlahi": "magahi_sarlahi",
    "Mahottari": "magahi_mahottari",
    "Dhanusha": "magahi_dhanusha",
    "Saptari": "magahi_saptari",
    "Morang": "magahi_morang",
}


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    # The source's combining apical mark is positioned far enough from the
    # following glyph that table extraction inserts a spurious word space.
    value = value.replace("̺ ", "̺")
    return unicodedata.normalize("NFC", value)


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int]]:
    """Return (concept number, gloss, lect, source form, printed page)."""
    concept_rows: list[tuple[int, str, list[str | None], int]] = []

    with pdfplumber.open(pdf_path) as document:
        # Zero-based 123:132 = PDF pages 124--132 = printed pages 113--121.
        for page_index in range(123, 132):
            tables = document.pages[page_index].extract_tables()
            if len(tables) != 1:
                raise ValueError(
                    f"Expected one wordlist table on PDF page {page_index + 1}, "
                    f"got {len(tables)}"
                )
            for row in tables[0]:
                if len(row) != 9:
                    raise ValueError(
                        f"Unexpected Magahi table width {len(row)} on "
                        f"PDF page {page_index + 1}: {row}"
                    )
                # Most rows put the number in column 2; a few duplicated table
                # cells put it in both columns 1 and 2.
                number = "".join((row[1] or row[0] or "").splitlines())
                number = number.strip().rstrip(".")
                if not number.isdigit():
                    if page_index == 123 and number == "S.N":
                        continue
                    if any(value for value in row):
                        raise ValueError(
                            f"Unnumbered wordlist row on PDF page {page_index + 1}: {row}"
                        )
                    continue
                concept_rows.append(
                    (int(number), _clean(row[3] or ""), list(row[4:9]), page_index - 10)
                )

    concepts = [row[0] for row in concept_rows]
    if concepts != list(range(1, 211)):
        raise ValueError(f"Expected concepts 1--210 in order, got {concepts}")

    records: list[tuple[int, str, str, str, int]] = []
    for concept, gloss, cells, printed_page in concept_rows:
        if not gloss:
            raise ValueError(f"Concept {concept} has no English gloss")
        for label, cell in zip(LECTS, cells, strict=True):
            form = _clean(cell or "")
            if not form or form == "-":
                raise ValueError(f"Concept {concept} is blank at {label}")
            records.append((concept, gloss, LECTS[label], form, printed_page))

    if len(records) != 1050:
        raise ValueError(f"Expected 1,050 printed forms, got {len(records)}")
    return records


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, lect, form, printed_page in extracted:
        occurrence[(concept, lect)] += 1
        rows.append(
            [
                lect,
                "",
                form,
                gloss,
                "",
                form,
                "",
                f"{SOURCE}[p. {printed_page}]",
                "",
                "",
                f"magahi:{concept}:{lect.removeprefix('magahi_')}:"
                f"{occurrence[(concept, lect)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
