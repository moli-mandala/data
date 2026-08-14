"""Extract the two Naaba target wordlists from Swenson's clean PDF.

Appendix D.5 (PDF pages 75--89, printed pages 67--81) contains a
325-item landscape comparison table.  Only Pibu and Kimathanka Naaba are
retained; all neighboring-language comparison columns are excluded.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "naaba" / "source.pdf"
OUTPUT = HERE.parent / "20260813-naaba.csv"
SOURCE = "swenson2025naaba"
LECTS = (("naaba_pibu", 210, 277), ("naaba_kimathanka", 277, 338))
NONFORMS = {"", "No entry", "Nepali"}


def _cell(words: list[tuple], left: float, right: float, top: float, bottom: float) -> str:
    selected = sorted(
        (y, x, text)
        for x, y, _x1, _y1, text, *_rest in words
        if left <= x < right and top - 1 <= y < bottom - 1
    )
    return " ".join(text for _y, _x, text in selected).strip()


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int]]:
    """Return (concept number, gloss, lect, form, printed page)."""
    document = fitz.open(pdf_path)
    concepts: dict[int, str] = {}
    records: list[tuple[int, str, str, str, int]] = []

    for page_index in range(74, 89):
        words = document[page_index].get_text("words")
        starts = sorted(
            (y, int(text))
            for x, y, _x1, _y1, text, *_rest in words
            if 70 < x < 95 and text.isdigit() and 1 <= int(text) <= 325
        )
        for row_index, (top, concept) in enumerate(starts):
            bottom = starts[row_index + 1][0] if row_index + 1 < len(starts) else 1000
            gloss = _cell(words, 100, 170, top, bottom)
            if not gloss:
                raise ValueError(f"Missing gloss for concept {concept}")
            concepts[concept] = gloss

            for lect, left, right in LECTS:
                cell = _cell(words, left, right, top, bottom)
                if cell in NONFORMS:
                    continue
                # This cell includes an English disambiguator inside the
                # transcription column; retain only the actual IPA token.
                if cell == "(ɲe̤ ‘barley’)":
                    cell = "ɲe̤"
                for form in re.split(r"\s*(?:,|~)\s*", cell):
                    # The embedded font maps two visibly IPA <i> glyphs to
                    # Cyrillic nje in its ToUnicode table.
                    form = form.replace("Њ", "i")
                    if form:
                        records.append((concept, gloss, lect, form, page_index - 7))

    expected_concepts = set(range(1, 326))
    if set(concepts) != expected_concepts:
        raise ValueError(
            f"Unexpected concepts: missing {sorted(expected_concepts - set(concepts))}, "
            f"extra {sorted(set(concepts) - expected_concepts)}"
        )
    if len(records) != 665:
        raise ValueError(f"Expected 665 target forms, got {len(records)}")
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
                f"naaba:{concept}:{lect.removeprefix('naaba_')}:{occurrence[(concept, lect)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
