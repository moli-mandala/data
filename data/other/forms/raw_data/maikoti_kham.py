"""Extract the four Maikoti Kham target lists from Leman (2020)."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pdfplumber


HERE = Path(__file__).resolve().parent
PDF = HERE.parents[3] / "tmp" / "pdfs" / "maikoti-kham" / "official.pdf"
OUTPUT = HERE.parent / "20260813-maikoti-kham.csv"
SOURCE = "leman2020maikoti"
LECTS = {
    "Maikot": (337, 416, "maikoti_maikot"),
    "Ranma": (416, 481, "maikoti_ranma"),
    "Arjal": (481, 551, "maikoti_arjal"),
    "Hukam": (551, 631, "maikoti_hukam"),
}


def _column(words, left, right, top, bottom):
    result = [w for w in words if left <= w["x0"] < right and top <= w["top"] < bottom]
    return sorted(result, key=lambda w: (round(w["top"], 1), w["x0"]))


def _join(words, *, compact=False):
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
            text += (" " if not compact or gap > 2 else "") + current["text"]
        rendered.append(text)
    return ("" if compact else " ").join(rendered).strip()


def extract(pdf_path=PDF):
    records = []
    numbers = []
    variants = Counter()
    with pdfplumber.open(pdf_path) as document:
        # PDF pages 44--56 (printed pages 38--50) contain Appendix A.4.
        for page_index in range(43, 56):
            words = document.pages[page_index].extract_words(
                x_tolerance=1, y_tolerance=2, keep_blank_chars=False
            )
            anchors = []
            for word in _column(words, 70, 82, 75, 594):
                if re.fullmatch(r"\d{1,3}", word["text"]):
                    number = int(word["text"])
                    if 1 <= number <= 325:
                        anchors.append((number, word["top"]))
            for index, (number, top) in enumerate(anchors):
                bottom = anchors[index + 1][1] if index + 1 < len(anchors) else 594
                numbers.append(number)
                gloss = _join(_column(words, 98, 201, top - 1, bottom - 1))
                for label, (left, right, lect) in LECTS.items():
                    cell = _join(_column(words, left, right, top - 1, bottom - 1), compact=True)
                    # Two i glyphs have broken ToUnicode mappings; both were
                    # checked against the rendered official PDF.
                    if (number, label) == (41, "Arjal"):
                        cell = "t͡ʃĩ"
                    elif (number, label) == (271, "Arjal"):
                        cell = "bĩ̤"
                    cell = cell.replace("dz͡", "d͡z").replace("dʒ͡", "d͡ʒ")
                    if not cell or cell in {"DK", "NA", "(N)"}:
                        continue
                    for form in (value.strip() for value in re.split(r"[/,]", cell)):
                        if not form:
                            continue
                        variants[(number, label)] += 1
                        records.append((number, gloss, label, lect, form, page_index - 5,
                                        variants[(number, label)]))
    if numbers != list(range(1, 290)):
        raise ValueError(f"Unexpected source item sequence: {numbers}")
    if any("�" in row[4] for row in records):
        raise ValueError(f"Replacement character in target transcription: {[row for row in records if '�' in row[4]]}")
    return records


def main():
    rows = []
    for number, gloss, label, lect, form, page, variant in extract():
        rows.append([lect, "", form, gloss, "", form, "", f"{SOURCE}[p. {page}]",
                     "", "", f"maikoti:{number}:{lect}:{variant}"])
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    print(f"Wrote {len(rows)} forms from {len(LECTS)} Maikoti target lects to {OUTPUT}")


if __name__ == "__main__":
    main()
