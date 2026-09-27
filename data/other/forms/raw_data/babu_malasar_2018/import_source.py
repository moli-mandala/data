"""Import Babu's 2018 Malasar numeral table from archived source HTML."""

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
HTML = PACKAGE / "Malasar.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-babu-malasar-numerals.csv"
SHA256 = "dbfdb263903805590c197b387c14170e2240479760c7d85d200f047865bf0c80"
SOURCE = "babu2018malasar"
NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 400, 800, 1000, 2000]
NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen", "twenty"]
GLOSSES = {**dict(enumerate(NAMES, 1)),
           **{n: "twenty-" + NAMES[n - 21] for n in range(21, 30)},
           30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
           80: "eighty", 90: "ninety", 100: "hundred", 400: "four hundred",
           800: "eight hundred", 1000: "thousand", 2000: "two thousand"}
LABEL = re.compile(r"(?<![0-9])([0-9][0-9 ]*)\s*\.")


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed archived Malasar HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 4 or len(tables[1].find_all("tr")) != 20:
        raise ValueError("Expected four HTML tables and twenty numeral rows")
    heading = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    credit = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    if "Malasar" not in heading or "Aswini Babu" not in credit or "201 8" not in credit:
        raise ValueError("Changed source lect or contributor attribution")
    units = []
    for row_index, row in enumerate(tables[1].find_all("tr"), 1):
        cells = row.find_all("td")
        if len(cells) != 2:
            raise ValueError(f"Expected two cells in row {row_index}")
        for column, cell in enumerate(cells, 1):
            joined = unicodedata.normalize("NFC", re.sub(r"\s+", " ", cell.get_text("", strip=False)).strip())
            labels = list(LABEL.finditer(joined))
            expected = 2 if (row_index, column) == (18, 2) else 1
            if len(labels) != expected:
                raise ValueError(f"Unexpected prompt labels at {row_index}:{column}: {joined!r}")
            for segment_index, match in enumerate(labels, 1):
                end = labels[segment_index].start() if segment_index < len(labels) else len(joined)
                form = joined[match.end():end].strip(" ,; ")
                number = int(match.group(1).replace(" ", ""))
                if not form:
                    raise ValueError(f"Blank Malasar form at numeral {number}")
                units.append({"number": number, "source_form": form, "printed_label": match.group(1),
                              "joined_cell": joined, "raw_markup": str(cell), "table_row": row_index,
                              "table_column": column, "segment_index": segment_index})
    if len(units) != 41 or sorted(u["number"] for u in units) != NUMBERS:
        raise ValueError("Unexpected Malasar prompt inventory")
    return sorted(units, key=lambda u: u["number"])


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        number = unit["number"]
        key = f"{SOURCE}:number:{number}"
        form = unit["source_form"]
        rows.append(["Malasar", "", form, GLOSSES[number], "", form, "",
                     f"{SOURCE}[Malasar table, numeral {number}]", "", "", key,
                     "", "", "", "num"])
        audit.append({**unit, "source_cell_key": key, "status": "ingested", "reason": "",
                      "language_id": "Malasar", "source_lect": "Malasar", "gloss": GLOSSES[number],
                      "entry_key": key,
                      "source_locator": f"HTML table 1, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "uncertainty": "", "review": "Printed 400 and 800 are independent prompts within one HTML cell." if number in {400, 800} else ""})
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit = generate()
    target = OUTPUT if args.install else DRAFT
    with target.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with AUDIT.open("w", encoding="utf-8") as stream:
        for item in audit:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"40 HTML cells, {len(audit)} printed prompts, {len(rows)} accepted rows")


if __name__ == "__main__":
    main()
