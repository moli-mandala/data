"""Extract the Unicode wordlist in Kondakov's 2013 Rabha survey.

The official SIL PDF is ``silesr2013_016.pdf``. Appendix B.3 (PDF pages
22--34, printed pages 18--30) gives 194 concepts for the Rongdani and
Maituri varieties of Rabha. The PDF was generated from Word and contains
real Unicode text; this extractor uses the fixed two-column layout rather
than OCR.

Download the PDF to ``tmp/pdfs/rabha/silesr2013_016.pdf`` before running.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "rabha" / "silesr2013_016.pdf"
OUTPUT = HERE.parent / "20260813-rabha.csv"
SOURCE = "kondakov2013rabha"

LECTS = {
    "Rongdani": "rabha_rongdani",
    "Maituri": "rabha_maituri",
}


def _page_spans(page: fitz.Page) -> list[dict]:
    spans = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans.extend(line["spans"])
    return spans


def _clean_form(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Word's font switching inserts spaces inside tied affricates and after
    # an unreleased stem-final stop; neither is printed as a word boundary.
    return text.replace("͡ ", "͡").replace("̚ d͡", "̚d͡")


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int]]:
    """Return (concept number, gloss, lect, form, printed page) records."""
    document = fitz.open(pdf_path)
    concepts: list[tuple[int, str, str, str, int]] = []
    concept_number = 0

    # Zero-based 21:34 = PDF pages 22--34 = printed pages 18--30.
    for page_index in range(21, 34):
        spans = _page_spans(document[page_index])
        for column_x in (72.0, 324.0):
            column = [
                span
                for span in spans
                if (span["origin"][0] < 300) == (column_x < 300)
            ]
            rongdani = sorted(
                (span for span in column if span["text"].strip() == "A. Rabha"),
                key=lambda span: span["origin"][1],
            )
            maituri = sorted(
                (span for span in column if span["text"].strip() == "B. Rabha"),
                key=lambda span: span["origin"][1],
            )
            if len(rongdani) != len(maituri):
                raise ValueError(f"Mismatched lect cells on PDF page {page_index + 1}")

            gloss_spans = [
                span
                for span in column
                if abs(span["origin"][0] - column_x) < 1
                and span["text"].strip()
                and span["text"].strip()
                not in {"A. Rabha", "B. Rabha", "Rongdani", "Maituri"}
                and 60 < span["origin"][1] < 730
            ]
            used_glosses: set[int] = set()
            cells = []
            for a_label, b_label in zip(rongdani, maituri):
                candidates = [
                    span
                    for span in gloss_spans
                    if span["origin"][1] < a_label["origin"][1] + 1
                    and id(span) not in used_glosses
                ]
                if not candidates:
                    raise ValueError(f"Missing gloss on PDF page {page_index + 1}")
                gloss = max(candidates, key=lambda span: span["origin"][1])
                used_glosses.add(id(gloss))
                cells.append((gloss, a_label, b_label))

            for cell_index, (gloss_span, a_label, b_label) in enumerate(cells):
                concept_number += 1
                gloss = gloss_span["text"].strip()
                top = gloss_span["origin"][1] - 2
                bottom = (
                    cells[cell_index + 1][0]["origin"][1] - 2
                    if cell_index + 1 < len(cells)
                    else 730
                )
                midpoint = (a_label["origin"][1] + b_label["origin"][1]) / 2

                for lect, low, high in (
                    ("Rongdani", top, midpoint),
                    ("Maituri", midpoint, bottom),
                ):
                    form_spans = [
                        span
                        for span in column
                        if span["origin"][0] >= column_x + 120
                        and low <= span["origin"][1] < high
                    ]
                    form_spans.sort(
                        key=lambda span: (round(span["origin"][1], 2), span["origin"][0])
                    )
                    form = _clean_form("".join(span["text"] for span in form_spans))
                    if not form:
                        raise ValueError(
                            f"Missing {lect} form for {gloss!r} on PDF page {page_index + 1}"
                        )
                    for alternate in (part.strip() for part in form.split(",")):
                        concepts.append(
                            (concept_number, gloss, lect, alternate, page_index - 3)
                        )

    if concept_number != 194:
        raise ValueError(f"Expected 194 concepts, got {concept_number}")
    if len(concepts) != 400:
        raise ValueError(f"Expected 400 printed forms, got {len(concepts)}")
    return concepts


def main() -> None:
    extracted = extract()
    occurrence: Counter[tuple[int, str]] = Counter()
    rows = []
    for concept, gloss, lect, form, printed_page in extracted:
        occurrence[(concept, lect)] += 1
        rows.append(
            [
                LECTS[lect],
                "",
                form,
                gloss,
                "",
                form,
                "",
                f"{SOURCE}[p. {printed_page}]",
                "",
                "",
                f"rabha:{concept}:{lect.lower()}:{occurrence[(concept, lect)]}",
            ]
        )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
