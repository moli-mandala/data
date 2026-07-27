"""Extract the omitted Drasi and Brokskat columns from Schmidt & Kaul (2008).

This is a reproducibility helper for the one-time OCR review.  Render PDF pages
28--52 at 300 dpi and run Tesseract's TSV output first, for example::

    pdftoppm -f 28 -l 52 -r 300 -png schmidt.pdf table
    for image in table-*.png; do
        tesseract "$image" "${image%.png}" --psm 6 -l eng tsv
    done

The output is deliberately a review file, not publication data: Schmidt's
specialist transcription must be checked against the page images before use.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROW_ALIASES = {"AQ.": 40, "44,": 44, "251.)": 251}


def words(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle, delimiter="\t") if row["level"] == "5"]


def row_markers(page_words: list[dict[str, str]]) -> list[tuple[int, int]]:
    markers = []
    for word in page_words:
        if int(word["left"]) >= 800:
            continue
        text = word["text"].strip()
        match = re.fullmatch(r"(\d{1,3})\.", text)
        number = int(match.group(1)) if match else ROW_ALIASES.get(text)
        if number and 1 <= number <= 297:
            markers.append((number, int(word["top"])))
    return sorted(set(markers), key=lambda item: item[1])


def column_text(
    page_words: list[dict[str, str]], top: int, bottom: int, left: int, right: int
) -> str:
    selected = []
    for word in page_words:
        x = int(word["left"]) + int(word["width"]) / 2
        y = int(word["top"])
        if top <= y < bottom and left <= x < right:
            selected.append((y, int(word["left"]), word["text"]))
    selected.sort()
    return " ".join(text for _, _, text in selected).strip()


def extract(directory: Path) -> list[dict[str, str | int]]:
    result = []
    for path in sorted(directory.glob("table-*.tsv")):
        page_words = words(path)
        markers = row_markers(page_words)
        for index, (number, top) in enumerate(markers):
            previous = markers[index - 1][1] if index else top - 80
            following = markers[index + 1][1] if index + 1 < len(markers) else top + 100
            row_top = (previous + top) // 2
            row_bottom = (top + following) // 2
            result.append(
                {
                    "Item": number,
                    "Gloss_OCR": re.sub(
                        r"^(?:\d{1,3}[.,)]?|AQ\.)\s*", "",
                        column_text(page_words, row_top, row_bottom, 450, 850),
                    ),
                    "Drasi_OCR": column_text(page_words, row_top, row_bottom, 1940, 2245),
                    "Brokskat_OCR": column_text(page_words, row_top, row_bottom, 2245, 2850),
                    "Page": int(re.search(r"(\d+)$", path.stem).group(1)),
                }
            )
    by_item = {int(row["Item"]): row for row in result}
    missing = set(range(1, 298)) - set(by_item)
    if missing:
        raise ValueError(f"OCR did not recognize table rows: {sorted(missing)}")
    return [by_item[number] for number in range(1, 298)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = extract(args.tsv_dir)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
