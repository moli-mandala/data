"""Attach only unambiguous exact-form etymologies in the Backstrom wordlists.

An unresolved row is linked when an already-curated Backstrom row has the same
gloss and the same NFC-normalized phonemic form, ignoring case and stress marks,
and every matching curated row points to one Parameter_ID.  This deliberately
does not use edit distance, broad sound correspondences, or gloss alone.

The generated review CSV records every change and an existing form that supports
it.  Re-running the script is safe: rows that already have an etymology are left
untouched.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
FORMS = HERE.parent / "20230416-northern.csv"
REVIEW = HERE / "northern_obvious_etymologies.csv"
STRESS = str.maketrans("", "", "ˈˌ")


def normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().lower().translate(STRESS)


def key(row: list[str]) -> tuple[str, str]:
    phonemic = row[5] or row[2]
    return normalized(phonemic), normalized(row[3])


def main() -> None:
    with FORMS.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    analyses: dict[tuple[str, str], set[str]] = defaultdict(set)
    evidence: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        if not row[1]:
            continue
        match_key = key(row)
        analyses[match_key].add(row[1])
        evidence.setdefault(
            (match_key[0], match_key[1], row[1]),
            [row[0], row[2], row[3]],
        )

    changes: list[list[str]] = []
    for row_number, row in enumerate(rows, 1):
        if row[1]:
            continue
        match_key = key(row)
        candidates = analyses.get(match_key, set())
        if len(candidates) != 1:
            continue
        parameter_id = next(iter(candidates))
        support = evidence[(match_key[0], match_key[1], parameter_id)]
        row[1] = parameter_id
        changes.append(
            [
                str(row_number), row[0], row[2], row[3], row[5], parameter_id,
                support[0], support[1],
                "same gloss + exact normalized phonemic form (stress ignored)",
            ]
        )

    with FORMS.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    # Keep the review record from the linking pass on later idempotent runs.
    if changes or not REVIEW.exists():
        with REVIEW.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Row", "Language_ID", "Form", "Gloss", "Phonemic",
                    "Parameter_ID", "Evidence_Language_ID", "Evidence_Form", "Basis",
                ]
            )
            writer.writerows(changes)

    print(f"Linked {len(changes)} obvious forms in {FORMS}")
    if changes:
        print(f"Wrote review details to {REVIEW}")
    else:
        print(f"No changes; preserved existing review details in {REVIEW}")


if __name__ == "__main__":
    main()
