"""Extract the four target wordlists from Rai, Rai & Thokar (2015).

Annex B of the born-digital survey PDF (PDF pages 113--123, printed
pages 100--110) contains 211 numbered prompts for Phedi, Khartamchha,
Patheka, and Baspani.  The table is real positioned text, not a scan.
Two legacy-font combining glyphs lack ToUnicode mappings; their visible
glyphs are decoded as tilde and underdot before the source transcription
is written to ``Phonemic``.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[4] / "tmp" / "pdfs" / "sampang" / "source.pdf"
OUTPUT = HERE.parent / "20260813-sampang.csv"
SOURCE = "rai-rai-thokar2015sampang"

LECTS = {
    "Phedi": "sampang_phedi",
    "Khartamchha": "sampang_khartamchha",
    "Patheka": "sampang_patheka",
    "Baspani": "sampang_baspani",
}

LEGACY_GLYPHS = {
    "(cid:1)": "\N{COMBINING TILDE}",
    "(cid:7)": "\N{COMBINING DOT BELOW}",
}


def _clean(value: str) -> str:
    value = value.replace("\n", " ")
    for legacy, unicode_value in LEGACY_GLYPHS.items():
        value = value.replace(legacy, unicode_value)
    value = re.sub(r"\s+([\u0300-\u036f])", r"\1", value)
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", value).strip())


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int]]:
    """Return (concept number, gloss, lect, source form, printed page)."""
    records: list[tuple[int, str, str, str, int]] = []
    concepts: list[int] = []

    with pdfplumber.open(pdf_path) as document:
        # Zero-based 112:123 = PDF pages 113--123 = printed pages 100--110.
        for page_index in range(112, 123):
            tables = document.pages[page_index].extract_tables()
            if len(tables) != 1:
                raise ValueError(
                    f"Expected one wordlist table on PDF page {page_index + 1}, "
                    f"got {len(tables)}"
                )
            for row in tables[0]:
                if len(row) != 6:
                    raise ValueError(
                        f"Expected six columns on PDF page {page_index + 1}: {row}"
                    )
                number = (row[0] or "").replace("\n", "").strip()
                if not number.isdigit():
                    # The first page has a single column-header row.
                    if page_index == 112 and row[1:] == [
                        "Word", "Phedi", "Khartamchha", "Patheka", "Baspani"
                    ]:
                        continue
                    raise ValueError(
                        f"Unnumbered wordlist row on PDF page {page_index + 1}: {row}"
                    )

                concept = int(number)
                concepts.append(concept)
                gloss = _clean(row[1] or "")
                # The source leaves prompt 146's English cell blank. Its
                # position between "light" and "below" in the standard list,
                # and the parallel elicitation form, identify it as "above".
                if concept == 146 and not gloss:
                    gloss = "above"
                if not gloss:
                    raise ValueError(f"Concept {concept} has no English gloss")

                printed_page = page_index - 12
                for label, cell in zip(LECTS, row[2:], strict=True):
                    cell = _clean(cell or "")
                    if not cell or cell == "-":
                        continue
                    # Slashes explicitly delimit alternate responses in the
                    # printed table. Parenthesized material stays with its form.
                    for form in (_clean(part) for part in re.split(r"[/,]", cell)):
                        if form:
                            records.append(
                                (concept, gloss, LECTS[label], form, printed_page)
                            )

    if concepts != list(range(1, 212)):
        raise ValueError(
            f"Expected concepts 1--211 in order, got {len(concepts)} rows"
        )
    if any("(cid:" in form for _c, _g, _l, form, _p in records):
        raise ValueError("Unmapped legacy font glyph remains in a source form")
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
                f"sampang:{concept}:{lect.removeprefix('sampang_')}:"
                f"{occurrence[(concept, lect)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
