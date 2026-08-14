"""Extract the five target wordlists from Rai, Rai & Thokar (2014).

Annex E of the born-digital survey PDF (PDF pages 131--137, printed
pages 116--122) contains 194 numbered prompts for Barbhanjyang, Gairi,
Pakha, Pokla, and Suke-ahal.  The table is positioned Unicode text, not
a scan.  Prompt 18 is explicitly blank for every site.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[4] / "tmp" / "pdfs" / "chhulung" / "source.pdf"
OUTPUT = HERE.parent / "20260813-chhulung.csv"
SOURCE = "rai-rai-thokar2014chhulung"

LECTS = {
    "Barbhanjyang": "chhulung_barbhanjyang",
    "Gairi": "chhulung_gairi",
    "Pakha": "chhulung_pakha",
    "Pokla": "chhulung_pokla",
    "Suke-ahal": "chhulung_suke_ahal",
}


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    # The combining dental mark is positioned far enough from the following
    # vowel that table extraction inserts a spurious space in hərd̪i.
    value = value.replace("d̪ i", "d̪i")
    return unicodedata.normalize("NFC", value)


def _normalized_row(row: list[str | None]) -> list[str | None]:
    """Return number, English gloss, and the five target cells."""
    if len(row) == 12:
        # The first page has duplicated/merged cells around its header.
        return [row[0], row[3], *row[7:12]]
    if len(row) == 8:
        return [row[0], row[1], *row[3:8]]
    raise ValueError(f"Unexpected Chhulung table width {len(row)}: {row}")


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int]]:
    """Return (concept number, gloss, lect, source form, printed page)."""
    concept_rows: list[tuple[int, str, list[str | None], int]] = []

    with pdfplumber.open(pdf_path) as document:
        # Zero-based 130:137 = PDF pages 131--137 = printed pages 116--122.
        for page_index in range(130, 137):
            tables = document.pages[page_index].extract_tables()
            if len(tables) != 1:
                raise ValueError(
                    f"Expected one wordlist table on PDF page {page_index + 1}, "
                    f"got {len(tables)}"
                )
            for raw_row in tables[0]:
                row = _normalized_row(raw_row)
                number = "".join((row[0] or "").splitlines()).strip().rstrip(".")
                if not number.isdigit():
                    # Only the first page's title/header row is unnumbered.
                    if page_index == 130:
                        continue
                    if any(value for value in row):
                        raise ValueError(
                            f"Unnumbered wordlist row on PDF page {page_index + 1}: {row}"
                        )
                    continue
                concept_rows.append(
                    (int(number), _clean(row[1] or ""), list(row[2:]), page_index - 14)
                )

    concepts = [row[0] for row in concept_rows]
    if concepts != list(range(1, 195)):
        raise ValueError(f"Expected concepts 1--194 in order, got {concepts}")

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
            # A slash marks explicit alternatives. Line breaks within all
            # other cells are source multiword expressions and remain spaces.
            for form in (_clean(part) for part in cell.split("/")):
                if form:
                    records.append(
                        (concept, gloss, LECTS[label], form, printed_page)
                    )
        if not populated:
            blank_concepts.add(concept)

    if blank_concepts != {18}:
        raise ValueError(f"Unexpected all-site blank prompts: {blank_concepts}")
    if len(records) != 970:
        raise ValueError(f"Expected 970 printed forms, got {len(records)}")
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
                f"chhulung:{concept}:{lect.removeprefix('chhulung_')}:"
                f"{occurrence[(concept, lect)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
