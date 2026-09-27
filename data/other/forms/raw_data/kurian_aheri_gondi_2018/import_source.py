"""Import Kurian's archived 2018 Aheri Gondi numeral table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup


PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parents[4]
HTML = PACKAGE / "Gondi-Aheri.htm"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-kurian-aheri-gondi-numerals.csv"
SHA256 = "d7f14d4b9f6e55674a8ba00aefb2d89963318ff3901eda3d2bdfb45e1b9ad225"
SOURCE = "kurian2018aherigondi"
NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1000, 2000]
NAMES = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty",
]
GLOSSES = {
    **dict(enumerate(NAMES, 1)),
    **{n: "twenty-" + NAMES[n - 21] for n in range(21, 30)},
    30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    80: "eighty", 90: "ninety", 100: "hundred", 200: "two hundred",
    400: "four hundred", 800: "eight hundred", 1000: "thousand", 2000: "two thousand",
}
LABEL = re.compile(r"(?<![0-9])([0-9][0-9 ]*)\s*\.")


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed Aheri Gondi archive HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 4 or len(tables[1].find_all("tr")) != 20:
        raise ValueError("Expected four tables and twenty numeral rows")
    if "Aheri Gondi" not in tables[0].get_text(" ", strip=True):
        raise ValueError("Changed language heading")
    credit = tables[2].get_text(" ", strip=True)
    if "Benny Kurian" not in credit or "September 7, 2018" not in credit:
        raise ValueError("Changed contributor attribution")
    units = []
    for row_index, row in enumerate(tables[1].find_all("tr"), 1):
        cells = row.find_all("td")
        if len(cells) != 2:
            raise ValueError(f"Expected two columns in row {row_index}")
        for column, cell in enumerate(cells, 1):
            # Adjacent inline spans can split one phonetic word without source spaces:
            # e.g. 3. muː</span><span>ɖ</span><span>u.
            joined = unicodedata.normalize("NFC", re.sub(r"\s+", " ", cell.get_text("", strip=False)).strip())
            labels = list(LABEL.finditer(joined))
            if len(labels) != (2 if (row_index, column) in {(17, 2), (18, 2)} else 1):
                raise ValueError(f"Unexpected label count at {row_index}:{column}: {joined!r}")
            for segment_index, match in enumerate(labels, 1):
                end = labels[segment_index].start() if segment_index < len(labels) else len(joined)
                form = joined[match.end():end].strip(" ,; ")
                if not form:
                    raise ValueError(f"Empty reading at {row_index}:{column}:{segment_index}")
                units.append({
                    "number": int(match.group(1).replace(" ", "")),
                    "printed_label": match.group(1),
                    "source_form": form,
                    "joined_cell": joined,
                    "raw_markup": str(cell),
                    "table_row": row_index,
                    "table_column": column,
                    "segment_index": segment_index,
                })
    if len(units) != 42 or sorted(u["number"] for u in units) != NUMBERS:
        raise ValueError("Unexpected Aheri Gondi prompt inventory")
    return sorted(units, key=lambda x: x["number"])


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        number = unit["number"]
        form = unit["source_form"]
        key = f"{SOURCE}:number:{number}"
        rows.append([
            "Aheri Gondi", "", form, GLOSSES[number], "", form, "",
            f"{SOURCE}[Aheri Gondi table, numeral {number}]", "", "", key,
            "", "", "", "num",
        ])
        audit.append({
            **unit,
            "source_cell_key": key,
            "status": "ingested",
            "reason": "",
            "language_id": "Aheri Gondi",
            "source_lect": "Aheri Gondi",
            "gloss": GLOSSES[number],
            "parsed_form": form,
            "entry_key": key,
            "source_locator": f"HTML table 1, row {unit['table_row']}, column {unit['table_column']}, segment {unit['segment_index']}, numeral {number}",
            "uncertainty": "",
            "review": "Adjacent spans joined without inserted spaces; explicit numbered labels split 100/200 and 400/800 cells.",
        })
    if len(rows) != 42 or len({r[10] for r in rows}) != 42:
        raise ValueError("Unexpected Aheri Gondi output")
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit = generate()
    print(f"{len(audit)} numbered source units, {len(rows)} installed readings")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
