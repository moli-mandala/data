"""Extract the six Unicode IPA wordlists in Swenson's Gurung survey.

Appendix A (PDF pages 41--70, printed pages 34--63) contains 325
numbered concepts. The PDF is born-digital; no OCR is used.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "gurung" / "silesr2019_010.pdf"
OUTPUT = HERE.parent / "20260813-gurung.csv"
SOURCE = "swenson2019gurung"
SITES = ("Ajirkot", "Pyarjung", "Maling", "Yangjakot", "Birethanti", "Bhurdumpola")
LECTS = {site: f"gurung_{site.lower()}" for site in SITES}
CONCEPT = re.compile(r"^(\d+)\.\s*(.*)$")


def extract(pdf_path: Path = PDF):
    document = fitz.open(pdf_path)
    lines = []
    for page_index in range(40, 70):
        printed_page = page_index - 6
        lines.extend(
            (text.strip(), printed_page)
            for text in document[page_index].get_text().splitlines()
            if text.strip() and text.strip() != str(printed_page)
        )

    starts = [(i, CONCEPT.match(text)) for i, (text, _) in enumerate(lines) if CONCEPT.match(text)]
    if [int(match.group(1)) for _, match in starts] != list(range(1, 326)):
        raise ValueError("Expected the consecutively numbered concepts 1--325")

    records = []
    for position, (start, match) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        chunk = lines[start:end]
        concept = int(match.group(1))
        # Wrapped English labels precede the first Devanagari line/site row.
        gloss_parts = [match.group(2).strip()]
        for text, _ in chunk[1:]:
            if text in SITES or any("\u0900" <= char <= "\u097f" for char in text):
                break
            gloss_parts.append(text)
        gloss = re.sub(r"\s+", " ", " ".join(gloss_parts)).strip()

        for i, (text, printed_page) in enumerate(chunk):
            if text not in SITES:
                continue
            following = [value for value, _ in chunk[i + 1 :] if value]
            if not following:
                raise ValueError(f"Missing value after {text}, concept {concept}")
            form = following[0]
            if form == "X" or form in SITES or CONCEPT.match(form):
                continue
            records.append((concept, gloss, text, form, printed_page))
    return records


def main():
    records = extract()
    occurrence = Counter()
    rows = []
    for concept, gloss, site, cell, printed_page in records:
        for alternate in re.split(r"\s*/\s*", cell):
            if not alternate:
                continue
            occurrence[(concept, site)] += 1
            rows.append([
                LECTS[site], "", alternate, gloss, "", alternate, "",
                f"{SOURCE}[p. {printed_page}]", "", "",
                f"gurung:{concept}:{site.lower()}:{occurrence[(concept, site)]}",
            ])
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(SITES)} lects to {OUTPUT}")


if __name__ == "__main__":
    main()
