#!/usr/bin/env python3
"""Fill missing base-language coordinates from their explicit dialect points.

The arithmetic centroid is a display coordinate, not a claim about a language's
origin or exact extent. Exact elicitation/locality coordinates remain in
``dialects.csv``. Existing base-language coordinates are never overwritten.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).parent


def mean(values: list[str]) -> str:
    value = sum(float(item.strip()) for item in values) / len(values)
    return f"{value:.6f}".rstrip("0").rstrip(".")


def backfill(
    languages_path: Path = ROOT / "cldf/languages.csv",
    dialects_path: Path = ROOT / "cldf/dialects.csv",
) -> list[tuple[str, int, str, str]]:
    with languages_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        languages = list(reader)
    with dialects_path.open(encoding="utf-8", newline="") as stream:
        dialects = list(csv.DictReader(stream))

    points: dict[str, list[dict[str, str]]] = defaultdict(list)
    for dialect in dialects:
        if dialect.get("Latitude") and dialect.get("Longitude"):
            points[dialect["Language_ID"]].append(dialect)

    changed = []
    for language in languages:
        if language.get("Latitude") and language.get("Longitude"):
            continue
        evidence = points.get(language["ID"], [])
        if not evidence:
            continue
        language["Latitude"] = mean([row["Latitude"] for row in evidence])
        language["Longitude"] = mean([row["Longitude"] for row in evidence])
        # A/B describe sourced points; C marks an inferred display centroid.
        language["Quality"] = language.get("Quality") or "C"
        changed.append(
            (language["ID"], len(evidence), language["Latitude"], language["Longitude"])
        )

    fd, temporary = tempfile.mkstemp(prefix=languages_path.name + ".", dir=languages_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(languages)
        os.replace(temporary, languages_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return changed


if __name__ == "__main__":
    changed = backfill()
    for language_id, count, latitude, longitude in changed:
        print(f"{language_id}: {latitude},{longitude} from {count} dialect point(s)")
    print(f"filled {len(changed)} base languages")
