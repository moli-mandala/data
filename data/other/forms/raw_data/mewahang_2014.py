"""Extract the five target wordlists from Rai, Rai & Thokar (2014).

Annex E of the born-digital survey PDF (PDF pages 136--141, printed
pages 122--127) contains 210 numbered prompts for Yaphu, Mangtewa,
Tamku, Bala, and Yamdang.  The table is positioned Unicode text, not a
scan.  Prompts 73, 176, and 210 are explicitly blank for every site.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[4] / "tmp" / "pdfs" / "mewahang" / "source.pdf"
OUTPUT = HERE.parent / "20260813-mewahang.csv"
SOURCE = "rai-rai-thokar2014mewahang"

LECTS = {
    "Yaphu": "eastern_mewahang_yaphu",
    "Mangtewa": "eastern_mewahang_mangtewa",
    "Tamku": "western_mewahang_tamku",
    "Bala": "western_mewahang_bala",
    "Yamdang": "western_mewahang_yamdang",
}


def _clean(value: str) -> str:
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", value).strip())


def _normalized_row(row: list[str | None]) -> list[str | None]:
    """Return number, English gloss, and the five target cells."""
    if len(row) == 11:
        # The first page has two duplicated/merged cells around its header.
        return [row[0], row[2], *row[4:9]]
    if len(row) == 8:
        return [row[0], row[1], *row[3:8]]
    raise ValueError(f"Unexpected Mewahang table width {len(row)}: {row}")


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int]]:
    """Return (concept number, gloss, lect, source form, printed page)."""
    concept_rows: list[list[object]] = []

    with pdfplumber.open(pdf_path) as document:
        # Zero-based 135:141 = PDF pages 136--141 = printed pages 122--127.
        for page_index in range(135, 141):
            tables = document.pages[page_index].extract_tables()
            if len(tables) != 1:
                raise ValueError(
                    f"Expected one wordlist table on PDF page {page_index + 1}, "
                    f"got {len(tables)}"
                )
            for raw_row in tables[0]:
                row = _normalized_row(raw_row)
                number = (row[0] or "").replace("\n", "").strip().rstrip(".")
                gloss = _clean(row[1] or "")

                if number.isdigit():
                    concept_rows.append(
                        [int(number), gloss, list(row[2:]), page_index - 13]
                    )
                    continue

                # Ignore the two title/header rows on the first page.
                if page_index == 135 and not number:
                    continue
                if number == "S.N":
                    continue

                # Prompt 108 wraps "younger brother" across the page break.
                if not number and gloss and concept_rows:
                    concept_rows[-1][1] = _clean(
                        f"{concept_rows[-1][1]} {gloss}"
                    )
                    continue
                if any(value for value in row):
                    raise ValueError(
                        f"Unnumbered wordlist row on PDF page {page_index + 1}: {row}"
                    )

    concepts = [row[0] for row in concept_rows]
    if concepts != list(range(1, 211)):
        raise ValueError(f"Expected concepts 1--210 in order, got {concepts}")

    records: list[tuple[int, str, str, str, int]] = []
    blank_concepts = set()
    for concept, gloss, cells, printed_page in concept_rows:
        if not gloss:
            raise ValueError(f"Concept {concept} has no English gloss")
        populated = False
        for label, cell in zip(LECTS, cells, strict=True):
            cell = _clean(cell or "")
            if not cell or cell == "-":
                continue
            populated = True
            for form in (_clean(part) for part in cell.split("/")):
                if form:
                    records.append(
                        (concept, gloss, LECTS[label], form, printed_page)
                    )
        if not populated:
            blank_concepts.add(concept)

    if blank_concepts != {73, 176, 210}:
        raise ValueError(f"Unexpected all-site blank prompts: {blank_concepts}")
    if len(records) != 1164:
        raise ValueError(f"Expected 1,164 printed forms, got {len(records)}")
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
                f"mewahang:{concept}:{lect.split('_')[-1]}:"
                f"{occurrence[(concept, lect)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
