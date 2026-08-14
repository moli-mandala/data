"""Extract the three target Kochila Tharu wordlists from Eichentopf and Mitchell (2020)."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "kochila-tharu" / "official.pdf"
OUTPUT = HERE.parent / "20260813-kochila-tharu.csv"
SOURCE = "eichentopf-mitchell2020kochila"
LECTS = {
    "Morang (East)": "kochila_morang_east",
    "Bara (West)": "kochila_bara_west",
    "Siraha (Central)": "kochila_siraha_central",
}
EXPECTED_COUNTS = {
    "Morang (East)": 279,
    "Bara (West)": 274,
    "Siraha (Central)": 280,
}


def _page_text(page) -> str:
    lines = (page.extract_text() or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.fullmatch(r"\d+", lines[0].strip()):
        lines.pop(0)
    text = "\n".join(lines)
    text = re.sub(
        r"^Number English Gloss Nepali Gloss IPA Variety Transcription Grouping\s*",
        "",
        text,
    )
    # These are explanatory rows/footnotes rather than lexical items.
    text = re.sub(r"(?ms)^197–210\b.*$", "", text)
    text = re.sub(r"(?ms)^a Between numbers\b.*$", "", text)
    return text


def _english_gloss(header: str, number: int) -> str:
    header = " ".join(header.split())
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9 ’'()/,\-–]+", header)
    if not match:
        raise ValueError(f"Could not extract English gloss from {header!r}")
    gloss = match.group(0).strip()
    # The source's footnote marker is extracted inline after item 239's gloss.
    return "go" if number == 239 and gloss == "goa" else gloss


def _clean_form(number: int, variety: str, cell: str) -> tuple[str, str]:
    notes = []
    if cell.endswith(" excluded"):
        cell = cell[: -len(" excluded")]
        notes.append("Excluded from the source lexical-similarity calculation")
    form = re.sub(r"\s+\d+(?:\s+\d+)*\s*$", "", cell).strip()
    if not form or form.startswith("not applicable") or form.startswith("(same as"):
        return "", "; ".join(notes)

    for label in ("see face", "see oil", "head", "most often", "daughter’s"):
        suffix = f" ({label})"
        if form.endswith(suffix):
            form = form[: -len(suffix)].strip()
            notes.append(f"Source note: {label}")

    if number == 122 and variety == "Morang (East)":
        # This single legacy-font glyph has a broken ToUnicode map. Visual
        # inspection of printed p. 39 shows superscript i plus nasalization.
        form = "tʃaⁱ̃t̪i"
        notes.append("Normalized a legacy-font superscript-i mapping from the visual source")
    return form, "; ".join(notes)


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, str, int, int, str]]:
    reader = PdfReader(pdf_path)
    records = []
    numbers = []
    variant_numbers: Counter[tuple[int, str]] = Counter()

    # PDF pages 29--63 (printed pages 24--58) contain Appendix A.4.
    for page_index in range(28, 63):
        text = _page_text(reader.pages[page_index])
        for concept in re.finditer(
            r"(?m)^\s*(\d+)\s+(.*?)(?=^\s*\d+\s+|\Z)", text, re.S
        ):
            number = int(concept.group(1))
            chunk = concept.group(2)
            numbers.append(number)
            cells = []
            for variety in LECTS:
                match = re.search(rf"{re.escape(variety)}\s+(.*)", chunk)
                if match:
                    cells.append(
                        (match.start(), variety, match.end(), match.group(1))
                    )

            first_cell = cells[0][0] if cells else len(chunk)
            gloss = _english_gloss(chunk[:first_cell], number)
            for index, (start, variety, end, first_line) in enumerate(cells):
                stop = cells[index + 1][0] if index + 1 < len(cells) else len(chunk)
                cell = " ".join([first_line, *chunk[end:stop].splitlines()])
                cell = " ".join(cell.split())
                form, note = _clean_form(number, variety, cell)
                if not form:
                    continue
                variants = [
                    value.strip()
                    for value in re.split(r"\s*/\s*|\s+or\s+|\s+and\s+", form)
                    if value.strip()
                ]
                for value in variants:
                    variant_numbers[(number, variety)] += 1
                    records.append(
                        (
                            number,
                            gloss,
                            variety,
                            value,
                            page_index - 4,
                            variant_numbers[(number, variety)],
                            note,
                        )
                    )

    expected_numbers = [
        *range(1, 197),
        *range(211, 240),
        *range(241, 260, 2),
        260,
        261,
        263,
        265,
        267,
        *range(269, 310, 2),
        310,
        312,
        *range(314, 326),
    ]
    if numbers != expected_numbers:
        raise ValueError(f"Unexpected source item sequence: {numbers}")
    counts = Counter(record[2] for record in records)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected target-form counts: {counts}")
    if any("�" in record[3] for record in records):
        raise ValueError("Unresolved legacy-font replacement character in a target form")
    return records


def main() -> None:
    rows = []
    for number, gloss, variety, form, page, variant, note in extract():
        rows.append(
            [
                LECTS[variety],
                "",
                form,
                gloss,
                "",
                form,
                note,
                f"{SOURCE}[p. {page}]",
                "",
                "",
                f"kochila:{number}:{LECTS[variety]}:{variant}",
            ]
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} target lects to {OUTPUT}")


if __name__ == "__main__":
    main()
