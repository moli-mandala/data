"""Extract Western and Central Magar target lists from Swenson (2024).

Appendix A.3 (PDF pages 70--87, printed pages 60--77) presents a
three-column, 325-item comparison table.  The twelve newly collected
Western and Central Magar sites are retained.  Mudhebas (``Mudh``), the
Eastern Magar comparison list reused from Hilty (2013), is excluded.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "magar-2024" / "source.pdf"
OUTPUT = HERE.parent / "20260813-magar-2024.csv"
SOURCE = "swenson2024magar"

LECTS = {
    "Lasa": "western_magar_lasargha",
    "Math": "western_magar_mathagadhi",
    "Jhok": "western_magar_jhokedi",
    "Silu": "central_magar_siluwa",
    "Mity": "central_magar_mityal",
    "Dhar": "central_magar_dhardh",
    "Inas": "central_magar_inaskot",
    "Mich": "central_magar_michhurlung",
    "Rhis": "central_magar_rhising",
    "Arkh": "central_magar_arkhala",
    "Raik": "central_magar_raikot",
    "Bhad": "central_magar_bhadari",
}
SITE_CODES = {*LECTS, "Mudh", "All"}
NONFORMS = {"", "N", "DK", "--"}


def _text_in_box(
    words: list[tuple], left: float, right: float, top: float, bottom: float
) -> str:
    selected = sorted(
        (y, x, text)
        for x, y, _x1, _y1, text, *_rest in words
        if left <= x < right and top <= y < bottom
    )
    return " ".join(text for _y, _x, text in selected).strip()


def _forms(cell: str) -> list[str]:
    cell = cell.strip()
    if cell in NONFORMS:
        return []

    # The report marks some transcribed Nepali loans as N (form).  Keep the
    # source transcription, while bare N means that no Magar response was
    # recorded and is therefore omitted.
    loan = re.fullmatch(r"N\s*\((.+)\)", cell)
    if loan:
        cell = loan.group(1).strip()

    return [
        form
        for form in re.split(r"\s*(?:,|~)\s*", cell)
        if form and form not in NONFORMS
    ]


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int]]:
    """Return (concept number, gloss, lect, raw IPA, printed page)."""
    document = fitz.open(pdf_path)
    concepts: dict[int, str] = {}
    records: list[tuple[int, str, str, str, int]] = []

    for page_index in range(69, 87):
        words = document[page_index].get_text("words")
        headers = sorted(
            (round(x, 1), y, int(match.group(1)))
            for x, y, _x1, _y1, text, *_rest in words
            if (match := re.fullmatch(r"(\d+)\)", text))
            and 1 <= int(match.group(1)) <= 325
        )
        column_starts = sorted({x for x, _y, _concept in headers})

        for x, top, concept in headers:
            same_column = sorted(y for hx, y, _c in headers if hx == x and y > top)
            bottom = same_column[0] if same_column else 800
            column_index = column_starts.index(x)
            right = (
                column_starts[column_index + 1] - 3
                if column_index + 1 < len(column_starts)
                else 580
            )

            labels = sorted(
                (y, text)
                for wx, y, _x1, _y1, text, *_rest in words
                if abs(wx - x) < 1 and top < y < bottom and text in SITE_CODES
            )
            if not labels:
                raise ValueError(
                    f"No site rows for concept {concept} on PDF page {page_index + 1}"
                )

            heading = _text_in_box(words, x, right, top, labels[0][0])
            gloss_match = re.search(r"‘([^’]+)’", heading)
            if not gloss_match:
                raise ValueError(
                    f"No gloss for concept {concept} on PDF page {page_index + 1}: {heading!r}"
                )
            gloss = gloss_match.group(1)
            concepts[concept] = gloss

            for label_index, (label_top, code) in enumerate(labels):
                cell_bottom = (
                    labels[label_index + 1][0] - 2
                    if label_index + 1 < len(labels)
                    else bottom - 2
                )
                # ``Mudh`` is the reused Eastern comparator.  ``All`` occurs
                # only with the value "Nepali", i.e. no Magar target form.
                if code in {"Mudh", "All"}:
                    continue
                # The form baseline can begin about half a point above its
                # site-code baseline, so offset row bounds slightly upward.
                cell = _text_in_box(
                    words, x + 25, right, label_top - 2, cell_bottom
                )
                for form in _forms(cell):
                    records.append(
                        (concept, gloss, LECTS[code], form, page_index - 9)
                    )

    expected_concepts = set(range(1, 326))
    if set(concepts) != expected_concepts:
        raise ValueError(
            f"Unexpected concepts: missing {sorted(expected_concepts - set(concepts))}, "
            f"extra {sorted(set(concepts) - expected_concepts)}"
        )
    if len(records) != 1769:
        raise ValueError(f"Expected 1769 target forms, got {len(records)}")
    return records


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, lect, form, printed_page in extracted:
        occurrence[(concept, lect)] += 1
        short_lect = lect.removeprefix("western_magar_").removeprefix(
            "central_magar_"
        )
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
                f"magar-2024:{concept}:{short_lect}:{occurrence[(concept, lect)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
