"""Extract the Unicode wordlists in de Vries's Humla Tibetan survey.

Appendix A.3 (PDF pages 59--76, printed pages 49--66) lays out 205
concepts in three repeated page columns. Each concept has seven village rows.
The PDF is born-digital and uses Charis SIL for the IPA; no OCR is used.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "humla" / "silesr2020_013.pdf"
OUTPUT = HERE.parent / "20260813-humla.csv"
SOURCE = "devries2020humla"
SITES = ("Til", "Muchu", "Yalbang", "Kermi", "Yakpa", "Bargaun", "Dojam")
LECTS = {site: f"humla_{site.lower()}" for site in SITES}
COLUMN_X = (69.1, 313.5, 557.9)


def _spans(page):
    return [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["text"].strip()
    ]


def extract(pdf_path: Path = PDF):
    document = fitz.open(pdf_path)
    concepts = []

    for page_index in range(58, 76):
        spans = _spans(document[page_index])
        columns = COLUMN_X if page_index < 75 else (COLUMN_X[0],)
        for column_x in columns:
            column = [s for s in spans if abs(s["bbox"][0] - column_x) < 2]
            til_rows = sorted(
                (s for s in column if s["text"].strip() == "Til"),
                key=lambda s: s["bbox"][1],
            )
            for til in til_rows:
                y = til["bbox"][1]
                gloss_parts = [
                    s["text"].strip()
                    for s in sorted(column, key=lambda s: s["bbox"][1])
                    if y - 35 < s["bbox"][1] < y - 2
                    and s["text"].strip() not in SITES
                    and "Annapurna" not in s["font"]
                ]
                gloss = re.sub(r"\s+", " ", " ".join(gloss_parts)).strip()
                forms = []
                for offset, site in enumerate(SITES):
                    site_y = y + offset * 13.7
                    form_spans = [
                        s
                        for s in spans
                        if abs(s["bbox"][1] - site_y) < 1.5
                        and column_x + 70 < s["bbox"][0] < column_x + 175
                        and "Charis" in s["font"]
                    ]
                    form = " ".join(
                        s["text"].strip()
                        for s in sorted(form_spans, key=lambda s: s["bbox"][0])
                    )
                    forms.append((site, re.sub(r"\s+", " ", form).strip()))
                concepts.append((gloss, forms, page_index - 9))

    if len(concepts) != 205:
        raise ValueError(f"Expected 205 concepts, got {len(concepts)}")
    if any(not gloss for gloss, _, _ in concepts):
        raise ValueError("Missing an English elicitation gloss")
    if any(len(forms) != 7 for _, forms, _ in concepts):
        raise ValueError("A concept does not have all seven village rows")
    return concepts


def main():
    rows = []
    occurrence = Counter()
    for concept, (gloss, forms, printed_page) in enumerate(extract(), 1):
        for site, cell in forms:
            if not cell or cell in {"-", "x"}:
                continue
            # Slash and parenthesized forms are printed alternatives (especially
            # the nonpast/past verb pairs in the final pages), not annotations.
            parenthetical = re.fullmatch(r"\s*(.*?)\s*\(\s*(.*?)\s*\)\s*", cell)
            alternatives = list(parenthetical.groups()) if parenthetical else re.split(r"\s*/\s*", cell)
            for alternate in alternatives:
                if not alternate:
                    continue
                occurrence[(concept, site)] += 1
                rows.append(
                    [
                        LECTS[site], "", alternate, gloss, "", alternate, "",
                        f"{SOURCE}[p. {printed_page}]", "", "",
                        f"humla:{concept}:{site.lower()}:{occurrence[(concept, site)]}",
                    ]
                )

    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(SITES)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
