#!/usr/bin/env python3
"""Attach cell-crop Vision output to the fixed Kullu transcription scaffold.

Vision output is retained verbatim as a review aid.  It is never promoted to
``Transcription`` by this script; the source-image review remains a separate,
manual step.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

MARKER = re.compile(r"^@@\t.*/(i\d{3}-[A-Z]{3}-pdf\d{3}\.png)$")


def read_vision(path: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    current: str | None = None
    observations: list[tuple[float, str, list[str]]] = []

    def finish() -> None:
        nonlocal current, observations
        if current is None:
            return
        observations.sort(key=lambda row: row[0])
        primary = " ".join(row[1] for row in observations)
        alternates = " || ".join(" | ".join(row[2]) for row in observations if row[2])
        result[current] = (primary, alternates)

    for line in path.read_text(encoding="utf-8").splitlines():
        marker = MARKER.match(line)
        if marker:
            finish()
            current = marker.group(1)
            observations = []
            continue
        if current is None or not line:
            continue
        fields = line.split("\t")
        if len(fields) >= 5:
            observations.append((float(fields[0]), fields[4], fields[5:]))
    finish()
    return result


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
        filename = Path(row["Cell_Image"]).name
        row["Raw_OCR"], row["OCR_Alternates"] = ocr.get(filename, ("", ""))
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
