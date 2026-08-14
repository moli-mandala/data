"""Extract Brianne Smith's five Pahari survey wordlists.

Appendix B.5 of the born-digital SIL PDF (PDF pages 50--96, printed
pages 39--85) contains a 325-item comparative wordlist.  The five Pahari
field sites are retained and the seven Newar comparison lists are excluded.

Download the PDF to ``tmp/pdfs/pahari/source.pdf`` before running.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "pahari" / "source.pdf"
OUTPUT = HERE.parent / "20260813-pahari.csv"
SOURCE = "smith2022pahari"

TARGET_LECTS = {
    "Salintar": "pahari_salintar",
    "Sikharpa": "pahari_sikharpa",
    "Maasdada": "pahari_maasdada",
    "Sakhatar": "pahari_sakhatar",
    "Jamune": "pahari_jamune",
}
CONTROL_LECTS = {
    "Kathmandu",
    "Patan",
    "Bhaktapur",
    "Pyangaun",
    "Balami",
    "Chitlang",
    "Dolakha",
}
LABELS = {
    **{f"{name} (pa)": name for name in TARGET_LECTS},
    **{f"{name} (ne)": name for name in CONTROL_LECTS},
}
CONCEPT_RE = re.compile(r"(\d+)\.\s+(.+)")
SOURCE_NOTE_RE = re.compile(r"\s+\((?:N|H|E|RR|DK|DC)\)$")
GLOSS_OVERRIDES = {
    235: "how (what is it like)",
    236: "how (to do something)",
    243: "come down (3S-PT)",
    247: "climb down (3S-PT)",
    250: "don’t bring (3S-PT)",
    316: "your (formal sing)",
    317: "your (formal plural)",
    319: "their (formal plural)",
    322: "he/she (formal sing)",
    324: "you (formal plural)",
}
NONFORMS = {
    "",
    "(N)",
    "(H)",
    "(E)",
    "(DK)",
    "(DC)",
    "DK",
    "DC",
    "NA",
    "No record",
    "-",
}


def _join_cell(lines: list[str]) -> str:
    """Reassemble the four cells that wrap onto a second PDF text line."""
    value = ""
    for line in lines:
        if value and not value.endswith("+"):
            value += " "
        value += line
    return value.strip()


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int]]:
    """Return (concept number, gloss, lect, form, printed page)."""
    document = fitz.open(pdf_path)
    cells: list[tuple[int, str, str, str, int]] = []
    concepts: dict[int, str] = {}
    concept_number: int | None = None
    current_lect: str | None = None
    current_page: int | None = None
    cell_lines: list[str] = []

    def flush_cell() -> None:
        nonlocal current_lect, current_page, cell_lines
        if current_lect is not None:
            if concept_number is None or current_page is None:
                raise ValueError("Pahari cell found before its concept or page")
            cells.append(
                (
                    concept_number,
                    concepts[concept_number],
                    current_lect,
                    _join_cell(cell_lines),
                    current_page,
                )
            )
        current_lect = None
        current_page = None
        cell_lines = []

    # Zero-based 49:96 = physical PDF pages 50--96 = printed pages 39--85.
    for page_index in range(49, 96):
        lines = [line.strip() for line in document[page_index].get_text().splitlines()]
        lines = [line for line in lines if line]
        if not lines or not lines[0].isdigit():
            raise ValueError(f"Missing printed page number on PDF page {page_index + 1}")
        printed_page = int(lines[0])

        for line in lines[1:]:
            concept_match = CONCEPT_RE.fullmatch(line)
            if concept_match:
                flush_cell()
                concept_number = int(concept_match.group(1))
                gloss = re.sub(r"\s*\(DC\)$", "", concept_match.group(2)).strip()
                gloss = GLOSS_OVERRIDES.get(concept_number, gloss)
                concepts[concept_number] = gloss
                continue

            if line in LABELS:
                flush_cell()
                label = LABELS[line]
                if label in TARGET_LECTS:
                    current_lect = label
                    current_page = printed_page
                continue

            if current_lect is not None:
                cell_lines.append(line)

    flush_cell()

    expected_concepts = set(range(1, 326))
    if set(concepts) != expected_concepts:
        raise ValueError(
            f"Unexpected concepts: missing {sorted(expected_concepts - set(concepts))}, "
            f"extra {sorted(set(concepts) - expected_concepts)}"
        )
    # Six explicitly DC concepts have no wordlist rows for any variety.
    if len(cells) != 1595:
        raise ValueError(f"Expected 1,595 printed Pahari cells, got {len(cells)}")

    records: list[tuple[int, str, str, str, int]] = []
    for concept, gloss, lect, cell, printed_page in cells:
        if cell in NONFORMS:
            continue
        for form in re.split(r"\s*/\s*", cell):
            form = SOURCE_NOTE_RE.sub("", form).strip()
            # Two PDF glyphs that are visibly IPA <i> have an erroneous Greek
            # nu Unicode mapping in the embedded font's ToUnicode table.
            form = form.replace("ν", "i")
            if form and form not in NONFORMS:
                records.append((concept, gloss, lect, form, printed_page))

    if len(records) != 1452:
        raise ValueError(f"Expected 1,452 target forms, got {len(records)}")
    return records


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, lect, form, printed_page in extracted:
        occurrence[(concept, lect)] += 1
        rows.append(
            [
                TARGET_LECTS[lect],
                "",
                form,
                gloss,
                "",
                form,
                "",
                f"{SOURCE}[p. {printed_page}]",
                "",
                "",
                f"pahari:{concept}:{lect.casefold()}:{occurrence[(concept, lect)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(TARGET_LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
