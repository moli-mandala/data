"""Extract the two target Kudiya wordlists from Joseph (2024).

The born-digital SIL PDF contains 207 numbered Unicode-IPA items for two
Kudiya sites plus Kodava and Malayalam comparison lists.  Only the Kudiya
lists (KDG/G1 and KDK/K1) are emitted.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "kudiya" / "source.pdf"
OUTPUT = HERE.parent / "20260813-kudiya.csv"
SOURCE = "joseph2024kudiya"
LECTS = {"KDG": "kudiya_g1", "KDK": "kudiya_k1"}


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int, str]]:
    pages = [page.extract_text() or "" for page in PdfReader(pdf_path).pages]
    records = []
    seen = {code: [] for code in LECTS}
    current: tuple[int, str] | None = None

    # PDF pages 17--25 (printed pages 11--19) contain Appendix A.
    for page_index in range(16, 25):
        for line in pages[page_index].splitlines():
            heading = re.match(r"^\s*(\d+)\.\s+(.+?)\s*$", line)
            if heading:
                current = (int(heading.group(1)), heading.group(2))
                continue
            cell = re.match(r"^(KDG|KDK)\s+(.+?)\s*$", line)
            if not cell or current is None:
                continue
            code, form = cell.groups()
            number, gloss = current
            seen[code].append(number)
            if form == "NA":
                continue
            note = ""
            if form == "eɳ:aⁱ":
                form = "eɳːaⁱ"
                note = "Normalized source colon as phonetic length mark"
            records.append((number, gloss, code, form, page_index - 5, note))

    expected = list(range(1, 208))
    if any(numbers != expected for numbers in seen.values()):
        raise ValueError(f"Expected items 1--207 for both Kudiya lists: {seen}")
    if len(records) != 409:
        raise ValueError(f"Expected 409 attested target forms, got {len(records)}")
    return records


def main() -> None:
    rows = []
    for number, gloss, code, form, page, note in extract():
        rows.append([
            LECTS[code], "", form, gloss, "", form, note,
            f"{SOURCE}[p. {page}]", "", "", f"kudiya:{number}:{code.lower()}:1",
        ])
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} target lects to {OUTPUT}")


if __name__ == "__main__":
    main()
