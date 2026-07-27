"""Build the reviewed Schmidt & Kaul (2008) Table 3 import."""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "schmidt_kashmiri.csv"
OUTPUT = HERE.parent / "20260725-schmidt-kashmiri.csv"

LANGUAGES = {
    "Kashmiri": "K",
    "Kishtawari": "kash",
    "Poguli": "pog",
    "Siraji": "sir",
}


def forms(cell: str):
    """Split alternatives, while retaining source phrases as single forms."""
    for form in cell.split(","):
        form = unicodedata.normalize("NFC", form.strip())
        if form and form.casefold() not in {"no data", "data no", "—", "-"}:
            yield form


def main() -> None:
    output = []
    with SOURCE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column, language_id in LANGUAGES.items():
                for form in forms(row[column]):
                    output.append(
                        [
                            language_id,
                            "",
                            form,
                            row["Gloss"],
                            "",
                            form,
                            f"Table 3 item {row['Item']}",
                            "schmidt",
                        ]
                    )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(output)


if __name__ == "__main__":
    main()
