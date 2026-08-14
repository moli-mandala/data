"""Extract the target wordlists from Webster's north Gorkha survey."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "north-gorkha" / "official.pdf"
OUTPUT = HERE.parent / "20260813-north-gorkha.csv"
SOURCE = "webster2022north-gorkha"

SECTIONS = (
    (
        "tibetan",
        range(30, 50),
        239,
        {
            "NuS": "nubri_sama",
            "NuL": "nubri_lho",
            "NuN": "nubri_namrung",
            "NuP": "nubri_prok",
            "TsC": "tsum_chekampar",
        },
    ),
    (
        "gorkha",
        range(50, 75),
        241,
        {
            "SGB": "southern_ghale_barpak",
            "SGK": "southern_ghale_kyaura",
            "SGL": "southern_ghale_laprak",
            "NGJ": "northern_ghale_jagat",
            "NGP": "northern_ghale_philim",
            "NGU": "northern_ghale_uiya",
            "NGK": "northern_ghale_khorla",
            "NGN": "northern_ghale_nyak",
            "KGB": "kutang_ghale_bihi",
            "KGC": "kutang_ghale_chyak",
            "KGR": "kutang_ghale_rana",
            "GrK": "eastern_gorkha_tamang_kashigaun",
            "GrJ": "eastern_gorkha_tamang_keraunja",
            "LTa": "western_tamang_lamagara",
        },
    ),
)

HEADER = re.compile(r"^(\d{1,3})\s+(.+?)\s*$")
ENTRY = re.compile(r"^([A-Za-z]{3})\s+(\S+)\s+(.+?)\s*$")
LEADING_COUNT = re.compile(r"^\d+\s+")


def _page_lines(page):
    """Return meaningful lines from the PDF's native text layer."""
    return [line.strip() for line in (page.extract_text() or "").splitlines() if line.strip()]


def _alternatives(value):
    """Split source alternatives and remove their optional respondent counts."""
    for part in value.split(" / "):
        part = LEADING_COUNT.sub("", part.strip())
        if part and part not in {"—", "---"}:
            yield part


def extract(pdf_path: Path = PDF):
    reader = PdfReader(pdf_path)
    records = []
    for section, page_indices, final_concept, lects in SECTIONS:
        expected = 1
        concept = None
        gloss = None
        concepts = []
        variants = Counter()
        for page_index in page_indices:
            for line in _page_lines(reader.pages[page_index]):
                # The printed heading for item 203 accidentally reads "fo you".
                # It occupies the normal heading position and item 204 follows it.
                if section == "gorkha" and expected == 203 and line == "fo you (sg. informal)":
                    number, candidate_gloss = 203, "you (sg. informal)"
                else:
                    match = HEADER.fullmatch(line)
                    if not match:
                        number = None
                    else:
                        number, candidate_gloss = int(match.group(1)), match.group(2)
                if number == expected:
                    concept, gloss = number, candidate_gloss
                    concepts.append((concept, gloss))
                    expected += 1
                    continue

                match = ENTRY.fullmatch(line)
                if concept is None or not match or match.group(1) not in lects:
                    continue
                code, count, value = match.groups()
                if count == "0":
                    continue
                for form in _alternatives(value):
                    variants[(concept, code)] += 1
                    records.append(
                        (
                            section,
                            concept,
                            gloss,
                            lects[code],
                            form,
                            page_index - 5,
                            variants[(concept, code)],
                        )
                    )
        if [number for number, _ in concepts] != list(range(1, final_concept + 1)):
            raise ValueError(f"Unexpected {section} concept sequence: {concepts}")
    return records


def main():
    rows = []
    for section, number, gloss, lect, form, page, variant in extract():
        rows.append(
            [
                lect,
                "",
                form,
                gloss,
                "",
                form,
                "",
                f"{SOURCE}[p. {page}]",
                "",
                "",
                f"north-gorkha:{section}:{number}:{lect}:{variant}",
            ]
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len({row[0] for row in rows})} target lects to {OUTPUT}")


if __name__ == "__main__":
    main()
