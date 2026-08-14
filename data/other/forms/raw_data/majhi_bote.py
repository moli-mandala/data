"""Extract the six target Majhi and Bote wordlists from Page (2024).

Appendix B of the born-digital SIL report prints four Majhi site lists and
two Bote site lists in Unicode IPA.  This importer emits all attested target
forms, including separately listed variants; there are no comparator lists.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "majhi-bote" / "source.pdf"
OUTPUT = HERE.parent / "20260813-majhi-bote.csv"
SOURCE = "page2024majhi-bote"
LECTS = {
    "Kunauri": "majhi_kunauri",
    "Gaikura": "majhi_gaikura",
    "Majhigau": "majhi_majhigau",
    "Pachuwar": "majhi_pachuwar",
    "Kawasoti": "bote_kawasoti",
    "Madi": "bote_madi",
}
EXPECTED_COUNTS = {
    "Kunauri": 264,
    "Gaikura": 269,
    "Majhigau": 268,
    "Pachuwar": 264,
    "Kawasoti": 264,
    "Madi": 261,
}


def _page_text(page) -> str:
    """Remove the running printed-page number before parsing the last cell."""
    lines = (page.extract_text() or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.fullmatch(r"\d+", lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines)


def _english_gloss(header: str) -> str:
    header = " ".join(header.split())
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9 ’'()/,\-–]+", header)
    if not match:
        raise ValueError(f"Could not extract English gloss from {header!r}")
    return re.sub(r"\s+x$", "", match.group(0)).strip()


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int, int]]:
    records = []
    numbers = []
    variant_numbers: Counter[tuple[int, str]] = Counter()
    reader = PdfReader(pdf_path)

    # PDF pages 49--70 (printed pages 41--62) contain B.4 Wordlists.
    for page_index in range(48, 70):
        text = _page_text(reader.pages[page_index])
        concepts = re.finditer(
            r"(?m)^\s*(\d+)\.\s(.*?)(?=^\s*\d+\.\s|\Z)", text, re.S
        )
        for concept in concepts:
            number = int(concept.group(1))
            chunk = concept.group(2)
            numbers.append(number)
            cells = []
            for site in LECTS:
                match = re.search(rf"(?m)^\s*{site}(?:\s+(.*))?$", chunk)
                if match:
                    cells.append((match.start(), site, match.end(), match.group(1) or ""))

            first_cell = cells[0][0] if cells else len(chunk)
            gloss = _english_gloss(chunk[:first_cell])
            for index, (start, site, end, first_line) in enumerate(cells):
                stop = cells[index + 1][0] if index + 1 < len(cells) else len(chunk)
                cell = " ".join([first_line, *chunk[end:stop].splitlines()])
                cell = " ".join(cell.split())
                if not cell or cell == "x":
                    continue

                # Discard lexical-similarity group labels (e.g. ``a b`` or
                # ``c, a``), then emit slash- and comma-separated variants.
                form = re.sub(r"\s+[a-f](?:[\s,]+[a-f])*\s*$", "", cell).strip(" ,")
                variants = [
                    value.strip()
                    for value in re.split(r"\s*/\s*|\s*,\s*", form)
                    if value.strip()
                ]
                for value in variants:
                    variant_numbers[(number, site)] += 1
                    records.append(
                        (
                            number,
                            gloss,
                            site,
                            value,
                            page_index - 7,
                            variant_numbers[(number, site)],
                        )
                    )

    expected_numbers = [*range(1, 197), *range(211, 326)]
    if numbers != expected_numbers:
        raise ValueError(f"Unexpected source item sequence: {numbers}")
    counts = Counter(record[2] for record in records)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected target-form counts: {counts}")
    return records


def main() -> None:
    rows = []
    for number, gloss, site, form, page, variant in extract():
        rows.append(
            [
                LECTS[site],
                "",
                form,
                gloss,
                "",
                form,
                "",
                f"{SOURCE}[p. {page}]",
                "",
                "",
                f"majhi-bote:{number}:{site.lower()}:{variant}",
            ]
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} target lects to {OUTPUT}")


if __name__ == "__main__":
    main()
