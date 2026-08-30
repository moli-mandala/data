#!/usr/bin/env python3
"""Attach macOS Vision OCR candidates to fixed Kullu item/site cells.

This output is a review aid only. The importer rejects any row that is not
explicitly marked as manually transcribed from the source image.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

PAGE = re.compile(r"^@@\t.*pdf(\d{3})\.png$")


def read_vision(path: Path) -> dict[tuple[int, int, int], list[tuple[float, str, str]]]:
    cells: dict[tuple[int, int, int], list[tuple[float, str, str]]] = defaultdict(list)
    page = None
    for line in path.read_text(encoding="utf-8").splitlines():
        marker = PAGE.match(line)
        if marker:
            page = int(marker.group(1))
            continue
        if page is None or not line:
            continue
        fields = line.split("\t")
        if len(fields) < 5:
            continue
        x, y, width, height = map(float, fields[:4])
        candidates = fields[4:]
        mid_x = x + width / 2
        mid_y = y + height / 2
        # Header writing sits above the horizontal rule. The centers of the
        # response rows are 200 + 100*n px from the top of the ~1770 px image.
        top_y = (1 - mid_y) * 1770
        row = round((top_y - 200) / 100)
        column = min(2, max(0, int(mid_x * 3)))
        if not 0 <= row < 16 or top_y < 145:
            continue
        cells[(page, column + 1, row)].append(
            (x, candidates[0], " | ".join(candidates[1:]))
        )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scaffold", type=Path)
    parser.add_argument("vision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ocr = read_vision(args.vision)
    with args.scaffold.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
        fields = list(rows[0])
    for row in rows:
        item = int(row["Item"])
        offset = (item - 1) % 16
        snippets = sorted(
            ocr.get((int(row["PDF_Page"]), int(row["Column"]), offset), [])
        )
        row["Raw_OCR"] = " ".join(snippet[1] for snippet in snippets)
        row["OCR_Alternates"] = " || ".join(
            snippet[2] for snippet in snippets if snippet[2]
        )
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"cells={len(rows)} ocr_nonblank={sum(bool(row['Raw_OCR']) for row in rows)} "
        f"pending={sum(row['Review'].startswith('pending') for row in rows)}"
    )


if __name__ == "__main__":
    main()
