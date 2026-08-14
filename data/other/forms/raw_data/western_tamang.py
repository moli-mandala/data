"""Extract the Unicode wordlist in Lipp's Western Tamang survey.

Appendix Table 14 of the born-digital appendices PDF (PDF pages 18--24,
printed pages 76--82) contains 280 concepts for Kashigaun, Jharlang, and
Sahugaun.  PyMuPDF recovers the ruled table directly; no OCR is used.

Download ``Western_Tamang_Survey_Appendices.pdf`` to
``tmp/pdfs/western-tamang`` before running.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = (
    HERE.parents[3]
    / "tmp"
    / "pdfs"
    / "western-tamang"
    / "Western_Tamang_Survey_Appendices.pdf"
)
OUTPUT = HERE.parent / "20260813-western-tamang.csv"
SOURCE = "lipp2014western-tamang"

SITES = ("Kashigaun", "Jharlang", "Sahugaun")
LECTS = {
    "Kashigaun": "eastern_gorkha_tamang_kashigaun",
    "Jharlang": "western_tamang_jharlang",
    "Sahugaun": "western_tamang_sahugaun",
}


def _clean_cell(text: str | None) -> str:
    if not text:
        return ""
    # A combining mark wrapped alone onto the next visual line still belongs
    # to the preceding segment. Other line wraps are ordinary spaces.
    text = re.sub(r"\n(?=[\u0300-\u036f])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_gloss(text: str | None) -> str:
    # The elicitation labels visibly use underscores as word separators.
    return re.sub(r"\s+", " ", _clean_cell(text).replace("_", " ")).strip()


def _clean_form(text: str) -> tuple[str, str]:
    text = _clean_cell(text)
    note = ""
    if text.endswith(" (short)"):
        text = text.removesuffix(" (short)")
        note = "Source marks this form as short."
    return text, note


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, str, int]]:
    """Return (concept, gloss, site, form, note, printed page) records."""
    document = fitz.open(pdf_path)
    records: list[tuple[int, str, str, str, str, int]] = []
    seen_concepts: list[int] = []

    # Zero-based 17:24 = PDF pages 18--24 = printed pages 76--82.
    for page_index in range(17, 24):
        tables = document[page_index].find_tables().tables
        if len(tables) != 1:
            raise ValueError(
                f"Expected one wordlist table on PDF page {page_index + 1}, got {len(tables)}"
            )
        for row in tables[0].extract():
            if not row[0] or not row[0].isdigit():
                continue
            if len(row) != 7:
                raise ValueError(f"Expected seven table cells, got {row}")
            concept = int(row[0])
            gloss = _clean_gloss(row[1])
            if not gloss:
                raise ValueError(f"Missing English gloss for item {concept}")
            seen_concepts.append(concept)

            for site, cell in zip(SITES, row[4:7]):
                cell = _clean_cell(cell)
                # The table uses 0/blank for two unelicited items and gives a
                # prose cross-reference rather than a form for Sahugaun 'she'.
                if (
                    not cell
                    or cell in {"0", "-"}
                    or cell.lower().startswith("not elicited")
                ):
                    continue
                form, note = _clean_form(cell)
                for alternate in re.split(r"\s*/\s*", form):
                    if alternate:
                        records.append(
                            (
                                concept,
                                gloss,
                                site,
                                alternate,
                                note,
                                page_index + 59,
                            )
                        )

    if seen_concepts != list(range(1, 281)):
        raise ValueError("Expected exactly the consecutively numbered items 1--280")
    if len(records) != 901:
        raise ValueError(f"Expected 901 printed forms/alternates, got {len(records)}")
    if Counter(site for _, _, site, _, _, _ in records) != {
        "Kashigaun": 296,
        "Jharlang": 307,
        "Sahugaun": 298,
    }:
        raise ValueError("Unexpected site totals")
    return records


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, site, form, note, printed_page in extracted:
        occurrence[(concept, site)] += 1
        rows.append(
            [
                LECTS[site],
                "",
                form,
                gloss,
                "",
                form,
                "",
                f"{SOURCE}[p. {printed_page}]",
                "",
                note,
                f"western-tamang:{concept}:{site.lower()}:{occurrence[(concept, site)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
