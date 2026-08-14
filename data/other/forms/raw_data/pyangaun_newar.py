"""Extract only the Pyangaun Newar column from Smith (2021)."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "pyangaun-newar" / "official.pdf"
OUTPUT = HERE.parent / "20260813-pyangaun-newar.csv"
SOURCE = "smith2021pyangaun"
LECT = "pyangaun_newar"

# Table columns on PDF pages 40--52.  The fourth variety column is Pyangaun;
# all seven surrounding columns are comparison lists and are deliberately ignored.
NUMBER_X = (62, 70)
GLOSS_X = (96, 184)
PYANGAUN_X = (400, 468)


def _words_in_column(words, bounds, top, bottom):
    left, right = bounds
    selected = [
        word
        for word in words
        if left <= word["x0"] < right and top <= word["top"] < bottom
    ]
    selected.sort(key=lambda word: (round(word["top"], 1), word["x0"]))
    return selected


def _join_words(words, *, compact: bool = False) -> str:
    if not words:
        return ""
    lines = []
    for word in words:
        top = round(word["top"], 1)
        if not lines or abs(lines[-1][0] - top) > 1.5:
            lines.append((top, [word]))
        else:
            lines[-1][1].append(word)
    rendered = []
    for _, parts in lines:
        text = parts[0]["text"]
        for previous, current in zip(parts, parts[1:]):
            gap = current["x0"] - previous["x1"]
            separator = " " if not compact or gap > 2 else ""
            text += separator + current["text"]
        rendered.append(text)
    # A phonetic cell that continues on a second line is a wrapped form, not a
    # second word. Genuine within-line word spaces are retained above.
    return ("" if compact else " ").join(rendered).strip()


def _clean_gloss(value: str) -> tuple[str, str]:
    value = " ".join(value.split())
    notes = []
    if value.endswith("(DC)"):
        value = value[: -len("(DC)")].strip()
        notes.append("Excluded from the source lexical-similarity calculation")
    return value, "; ".join(notes)


def extract(pdf_path: Path = PDF) -> list[tuple[int, str, str, int, int, str]]:
    records = []
    item_numbers = []
    variant_numbers: Counter[int] = Counter()

    with pdfplumber.open(pdf_path) as document:
        # PDF pages 40--52 are printed pages 32--44.
        for page_index in range(39, 52):
            words = document.pages[page_index].extract_words(
                x_tolerance=1, y_tolerance=2, keep_blank_chars=False
            )
            anchors = []
            for word in _words_in_column(words, NUMBER_X, 100, 780):
                if re.fullmatch(r"\d{1,3}", word["text"]):
                    number = int(word["text"])
                    if 1 <= number <= 325:
                        anchors.append((number, word["top"]))

            for index, (number, top) in enumerate(anchors):
                bottom = anchors[index + 1][1] if index + 1 < len(anchors) else 780
                if page_index == 39 and number == 19:
                    bottom = 520  # exclude the vowel-length footnote below the table
                item_numbers.append(number)
                gloss, note = _clean_gloss(
                    _join_words(_words_in_column(words, GLOSS_X, top - 1, bottom - 1))
                )
                cell = _join_words(
                    _words_in_column(words, PYANGAUN_X, top - 1, bottom - 1),
                    compact=True,
                )
                # Pdfplumber orders the tie bar after z in the source's voiced
                # alveolar affricate glyph runs; restore canonical IPA order.
                cell = cell.replace("dz͡", "d͡z")
                # These cells explicitly say that no Pyangaun form was supplied.
                if not cell or cell in {"(N)", "DK", "(Not used)", "NA"}:
                    continue
                nepali = False
                if cell.endswith(" (N)"):
                    cell = cell[: -len(" (N)")].strip()
                    nepali = True
                for form in (part.strip() for part in cell.split("/")):
                    if not form:
                        continue
                    variant_numbers[number] += 1
                    form_note = note
                    if nepali:
                        borrowing = "Marked as a Nepali borrowing in the source"
                        form_note = "; ".join(filter(None, [form_note, borrowing]))
                    records.append(
                        (
                            number,
                            gloss,
                            form,
                            page_index - 7,
                            variant_numbers[number],
                            form_note,
                        )
                    )

    if item_numbers != list(range(1, 326)):
        raise ValueError(f"Unexpected source item sequence: {item_numbers}")
    if any("�" in record[2] for record in records):
        raise ValueError("Replacement character in a target form")
    return records


def main() -> None:
    rows = []
    for number, gloss, form, page, variant, note in extract():
        rows.append(
            [
                LECT,
                "",
                form,
                gloss,
                "",
                form,
                note,
                f"{SOURCE}[p. {page}]",
                "",
                "",
                f"pyangaun:{number}:{variant}",
            ]
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} Pyangaun target forms to {OUTPUT}")


if __name__ == "__main__":
    main()
