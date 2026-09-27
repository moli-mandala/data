"""Import Kuriakose and Daniel's archived 2018 Muduga numeral table."""

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
HTML = PACKAGE / "Muduga.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-kuriakose-muduga-numerals.csv"
SHA256 = "3b240210f3bfa609450a010667030f9739bf02e8edc3f6030cbe30bd61a86b16"
SOURCE = "kuriakose2018muduga"
NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1000, 2000]
NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen", "twenty"]
GLOSSES = {**dict(enumerate(NAMES, 1)),
           **{n: "twenty-" + NAMES[n - 21] for n in range(21, 30)},
           30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
           80: "eighty", 90: "ninety", 100: "hundred", 200: "two hundred",
           400: "four hundred", 800: "eight hundred", 1000: "thousand", 2000: "two thousand"}
LABEL = re.compile(r"(?<![0-9])([0-9][0-9 ]*)\s*\.")


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed archived Muduga HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 4 or len(tables[1].find_all("tr")) != 20:
        raise ValueError("Expected four HTML tables and twenty numeral rows")
    heading = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    credit = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    if not ("Muduga" in heading and "Siby Kuriakose" in credit
            and "Stephen Daniel" in credit and "August 15" in credit and "201 8" in credit):
        raise ValueError("Changed Muduga lect or contributor attribution")
    units = []
    for row_index, row in enumerate(tables[1].find_all("tr"), 1):
        cells = row.find_all("td")
        if len(cells) != 2:
            raise ValueError(f"Expected two cells at row {row_index}")
        for column, cell in enumerate(cells, 1):
            joined = unicodedata.normalize("NFC", re.sub(r"\s+", " ", cell.get_text("", strip=False)).strip())
            labels = list(LABEL.finditer(joined))
            expected = 2 if (row_index, column) in {(17, 2), (18, 2)} else 1
            if len(labels) != expected:
                raise ValueError(f"Unexpected Muduga prompt labels at {row_index}:{column}: {joined!r}")
            for segment_index, match in enumerate(labels, 1):
                end = labels[segment_index].start() if segment_index < len(labels) else len(joined)
                number = int(match.group(1).replace(" ", ""))
                form = joined[match.end():end].strip(" ,，; ")
                if not form:
                    raise ValueError(f"Blank Muduga reading at {number}")
                units.append({"number": number, "source_form": form, "printed_label": match.group(1),
                              "joined_cell": joined, "raw_markup": str(cell), "table_row": row_index,
                              "table_column": column, "segment_index": segment_index})
    if len(units) != 42 or sorted(u["number"] for u in units) != NUMBERS:
        raise ValueError("Unexpected Muduga prompt inventory")
    return sorted(units, key=lambda u: u["number"])


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        number = unit["number"]
        key = f"{SOURCE}:number:{number}"
        form = unit["source_form"]
        rows.append(["Muduga", "", form, GLOSSES[number], "", form, "",
                     f"{SOURCE}[Muduga table, numeral {number}]", "", "", key,
                     "", "", "", "num"])
        audit.append({**unit, "source_cell_key": key, "status": "ingested", "reason": "",
                      "language_id": "Muduga", "source_lect": "Muduga", "gloss": GLOSSES[number],
                      "entry_key": key,
                      "source_locator": f"HTML table 1, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "uncertainty": "",
                      "review": "Adjacent inline spans joined without an invented space." if number in {15, 25} else ""})
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
    print(f"40 HTML cells, {len(audit)} printed prompts, {len(rows)} accepted readings")


if __name__ == "__main__":
    main()
