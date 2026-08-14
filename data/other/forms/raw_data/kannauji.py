"""Extract the target wordlists from John & Varghese's Kannauji survey.

Appendix C.3 of the born-digital SIL PDF (PDF pages 61--113, printed
pages 55--107) contains a 210-item comparative wordlist. Items 11, 23,
and 24 were deliberately omitted by the authors, leaving 207 concepts.
The thirteen field sites are retained; the Hindi, Bundeli, and Braj
Bhasha comparison lists are excluded.

Download the PDF to ``tmp/pdfs/kannauji/source.pdf`` before running.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "kannauji" / "source.pdf"
OUTPUT = HERE.parent / "20230526-kannauji.csv"
SOURCE = "kannauji"

TARGET_LECTS = {
    "Dehati-Sikandarpur",
    "Dehati-Badeli",
    "Hindi-Rohili",
    "Hindi-Sanayak",
    "Dehati-Kirkkichiyapur",
    "Hindi-Dhubar",
    "Kannauji-Central",
    "Hindi-Jamniya",
    "Hindi-Gohaniya",
    "Hindi-Gabchariyapur",
    "Hindi-Sarhati",
    "Hindi-Saraiyya",
    "Dehati-Madnapur",
}
CONTROL_LECTS = {"Hindi", "Bundeli", "Braj Bhasha"}
LABEL_ALIASES = {"Hincdi-Sarhati": "Hindi-Sarhati"}
CONCEPT_RE = re.compile(r"(\d+)\.\s+(.+)")


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, int, str, int]]:
    """Return (concept, gloss, lect, variant number, form, printed page)."""
    document = fitz.open(pdf_path)
    records: list[tuple[int, str, str, int, str, int]] = []
    concepts: set[int] = set()

    # Zero-based 60:113 = physical PDF pages 61--113.
    for page_index in range(60, 113):
        lines = [line.strip() for line in document[page_index].get_text().splitlines()]
        lines = [line for line in lines if line]
        if not lines or not lines[0].isdigit():
            raise ValueError(f"Missing printed page number on PDF page {page_index + 1}")
        printed_page = int(lines[0])
        concept_number = None
        gloss = None
        lect = None
        variant_number = None

        for line in lines[1:]:
            concept_match = CONCEPT_RE.fullmatch(line)
            if concept_match:
                concept_number = int(concept_match.group(1))
                gloss = concept_match.group(2).strip()
                concepts.add(concept_number)
                lect = None
                variant_number = None
                continue

            normalized_label = LABEL_ALIASES.get(line, line)
            if normalized_label in TARGET_LECTS | CONTROL_LECTS:
                lect = normalized_label
                variant_number = None
                continue

            if concept_number is not None and lect is not None and line.isdigit():
                variant_number = int(line)
                continue

            if variant_number is not None:
                # Code 0 is the literal table annotation "BY NAME", meaning
                # that no lexical response was elicited; it is not a form.
                if lect in TARGET_LECTS and variant_number != 0:
                    records.append(
                        (concept_number, gloss, lect, variant_number, line, printed_page)
                    )
                variant_number = None

    expected_concepts = set(range(1, 211)) - {11, 23, 24}
    if concepts != expected_concepts:
        raise ValueError(
            f"Unexpected concept inventory: missing {sorted(expected_concepts - concepts)}, "
            f"extra {sorted(concepts - expected_concepts)}"
        )
    if len(records) != 3033:
        raise ValueError(f"Expected 3,033 target forms, got {len(records)}")
    return records


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, lect, variant_number, form, printed_page in extracted:
        occurrence[(concept, lect)] += 1
        lect_key = re.sub(r"[^a-z0-9]+", "-", lect.casefold()).strip("-")
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
                f"kannauji:{concept}:{lect_key}:{occurrence[(concept, lect)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(TARGET_LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
