"""Audit and import Kapp's archived Alu Kurumba numeral table."""

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
HTML = PACKAGE / "Kurumba-Alu.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-kapp-alu-kurumba-numerals.csv"
SHA256 = "28a41dd60bd493765ffbdb63cd1d5d4603ba8d8ccd9e1e7d9c7e1995a67385d3"
SOURCE = "kapp1995alukurumba"
NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 200, 1000, 2000]
NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen", "twenty"]
GLOSSES = {**dict(enumerate(NAMES, 1)),
           **{n: "twenty-" + NAMES[n - 21] for n in range(21, 30)},
           30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
           80: "eighty", 90: "ninety", 100: "hundred", 200: "two hundred",
           1000: "thousand", 2000: "two thousand"}


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed archived Alu Kurumba HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 4 or len(tables[1].find_all("tr")) != 20:
        raise ValueError("Expected four tables and twenty numeral rows")
    heading = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    credit = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    if "Alu Kurumba" not in heading or "Dieter B. Kapp" not in credit or "1995" not in credit:
        raise ValueError("Changed source lect or contributor attribution")
    units = []
    for row_index, row in enumerate(tables[1].find_all("tr"), 1):
        cells = row.find_all("td")
        if len(cells) != 2:
            raise ValueError(f"Expected two cells in row {row_index}")
        for column, cell in enumerate(cells, 1):
            raw = re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
            match = re.fullmatch(r"([\d ]+)\s*\.\s*(.+)", raw)
            if not match:
                raise ValueError(f"Unparsed numeral at {row_index}:{column}: {raw!r}")
            number = int(match.group(1).replace(" ", ""))
            form = unicodedata.normalize("NFC", match.group(2).strip())
            units.append({"number": number, "printed_label": match.group(1),
                          "source_cell_key": f"{SOURCE}:number:{number}",
                          "raw_cell": raw, "raw_markup": str(cell), "source_form": form,
                          "table_row": row_index, "table_column": column})
    if len(units) != 40 or sorted(u["number"] for u in units) != NUMBERS:
        raise ValueError("Unexpected Alu Kurumba prompt inventory")
    return sorted(units, key=lambda item: item["number"])


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        number = unit["number"]
        key = unit["source_cell_key"]
        form = unit["source_form"]
        rows.append(["AluKurumba", "", form, GLOSSES[number], "", form, "",
                     f"{SOURCE}[Alu Kurumba table, numeral {number}]", "", "", key,
                     "", "", "", "num"])
        audit.append({**unit, "status": "ingested", "reason": "", "language_id": "AluKurumba",
                      "source_lect": "Alu Kurumba", "gloss": GLOSSES[number], "entry_key": key,
                      "source_locator": f"HTML table, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "review": "No etymon, donor, variant or site is inferred from the form or page."})
    if len(rows) != 40:
        raise ValueError("Unexpected Alu Kurumba answer count")
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
    print(f"{len(audit)} numbered cells, {len(rows)} accepted answers")


if __name__ == "__main__":
    main()
