#!/usr/bin/env python3
"""Apply checked-in transcription decisions to the generated review sheet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sheet", type=Path)
    parser.add_argument("--reviews", type=Path, default=HERE / "reviewed.tsv")
    args = parser.parse_args()
    with args.reviews.open(encoding="utf-8", newline="") as stream:
        reviews = {
            (row["Item"], row["Site"], row["Response"]): row
            for row in csv.DictReader(stream, delimiter="\t")
        }
    with args.sheet.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    seen = set()
    for row in rows:
        key = (row["Item"], row["Site"], row["Response"])
        if key not in reviews:
            continue
        review = reviews[key]
        row["Transcription"] = review["Transcription"]
        row["Review"] = review["Review"]
        row["Uncertainty"] = review["Uncertainty"]
        seen.add(key)
    missing = set(reviews) - seen
    if missing:
        raise SystemExit(f"review keys missing from scaffold: {sorted(missing)}")
    with args.sheet.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"applied {len(seen)} reviewed transcriptions")


if __name__ == "__main__":
    main()
