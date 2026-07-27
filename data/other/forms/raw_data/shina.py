"""Build the Schmidt & Kaul Shina form import.

``shina.csv`` is the original four-dialect transcription.  The source table's
previously omitted Drasi and Brokskat columns live in the separately reviewed
``schmidt_missing_dialects.csv`` so appending them does not renumber the 1,443
legacy rows referenced by the etymology audit.
"""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "shina.csv"
MISSING_DIALECTS = HERE / "schmidt_missing_dialects.csv"
ETYMOLOGY_REVIEW = HERE / "schmidt_shina_database_etymologies.csv"
OUTPUT = HERE.parent / "20230621-shina.csv"


def original_rows():
    with SOURCE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for language, value in row.items():
                if language in {"CDIAL", "Gloss"}:
                    continue
                for word in value.split(","):
                    word = word.strip()
                    notes = ""
                    if "(" in word:
                        word, notes = word.split("(", 1)
                        word = word.strip()
                        notes = notes.rstrip(")").strip()
                    if word:
                        yield [
                            language, row["CDIAL"], word, row["Gloss"], "",
                            word, notes, "schmidt",
                        ]


def missing_dialect_rows():
    with MISSING_DIALECTS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield [
                row["Language_ID"], "", row["Form"], row["Gloss"], "",
                row["Form"], row["Notes"], "schmidt",
            ]


def audited_original_rows():
    rows = list(original_rows())
    with ETYMOLOGY_REVIEW.open(encoding="utf-8") as handle:
        accepted = {
            int(row["Row"]): row["Parameter_ID"]
            for row in csv.DictReader(handle)
            if row["Decision"].strip().lower() in {"yes", "y", "accept", "accepted"}
        }
    for row_number, parameter_id in accepted.items():
        row = rows[row_number - 1]
        if row[1] and row[1] != parameter_id:
            raise ValueError(
                f"Row {row_number} links to {row[1]}, but the audit requires {parameter_id}"
            )
        row[1] = parameter_id
    return rows


def main() -> None:
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(audited_original_rows())
        writer.writerows(missing_dialect_rows())


if __name__ == "__main__":
    main()
