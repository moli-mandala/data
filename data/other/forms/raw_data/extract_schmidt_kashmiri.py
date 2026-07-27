"""Extract Table 3 of Schmidt & Kaul (2008) from reviewed OCR TSV files.

The table contains parallel Kashmiri, Kishtawari, Poguli, and Siraji columns.
This helper creates a rectangular review file; its output still needs checking
against the rendered PDF pages before it is used as source data.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROW_ALIASES = {
    "l.": "1",
    "59,": "59",
    "92.,": "92",
    "1021.": "102l",
    "251.I": "251",
    "251.1": "251",
}
COLUMNS = {
    "Gloss_OCR": (450, 850),
    "Kashmiri_OCR": (850, 1300),
    "Kishtawari_OCR": (1300, 1710),
    "Poguli_OCR": (1710, 2130),
    "Siraji_OCR": (2130, 2850),
}


def words(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle, delimiter="\t") if row["level"] == "5"]


def row_markers(page_words: list[dict[str, str]], scale: int) -> list[tuple[str, int]]:
    markers = []
    for word in page_words:
        if int(word["left"]) >= 800 * scale:
            continue
        text = word["text"].strip()
        match = re.fullmatch(r"(\d{1,3})([a-z]?)\.", text)
        key = f"{match.group(1)}{match.group(2)}" if match else ROW_ALIASES.get(text)
        if key and 1 <= int(re.match(r"\d+", key).group()) <= 267:
            markers.append((key, int(word["top"])))
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


def sort_key(key: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)([a-z]?)", key)
    return int(match.group(1)), match.group(2)


def extract(directory: Path) -> list[dict[str, str | int]]:
    result = []
    for path in sorted(directory.glob("kash-*-lat.tsv")):
        page_words = words(path)
        scale = 2 if max(int(word["left"]) for word in page_words) > 4000 else 1
        markers = row_markers(page_words, scale)
        for index, (key, top) in enumerate(markers):
            previous = markers[index - 1][1] if index else top - 80
            following = markers[index + 1][1] if index + 1 < len(markers) else top + 100
            row_top = (previous + top) // 2
            row_bottom = (top + following) // 2
            if key in {"102", "149"}:
                continue
            row: dict[str, str | int] = {"Key": key}
            for name, (left, right) in COLUMNS.items():
                text = column_text(
                    page_words, row_top, row_bottom, left * scale, right * scale
                )
                if name == "Gloss_OCR":
                    text = re.sub(r"^\d{1,3}[a-z]?[.,I]?\s*", "", text)
                row[name] = text
            row["Page"] = int(re.search(r"kash-(\d+)-lat$", path.stem).group(1))
            result.append(row)

    by_key = {str(row["Key"]): row for row in result}
    expected = {str(number) for number in range(1, 268)} - {"264"}
    expected |= {f"97{letter}" for letter in "abcdefg"}
    expected |= {f"102{letter}" for letter in "abcdefghijkl"}
    expected |= {"149a", "149b"}
    # The unlettered month and sheep rows introduce sublists and contain no forms.
    expected -= {"102", "149"}
    missing = expected - set(by_key)
    unexpected = set(by_key) - expected
    if missing or unexpected:
        raise ValueError(
            f"OCR row-key mismatch; missing={sorted(missing, key=sort_key)}, "
            f"unexpected={sorted(unexpected, key=sort_key)}"
        )
    return [by_key[key] for key in sorted(by_key, key=sort_key)]


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
