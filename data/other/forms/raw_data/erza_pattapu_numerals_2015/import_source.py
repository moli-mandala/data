"""Audit and import the complete Erza/BSI 2015 Pattapu numeral table."""

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
HTML = PACKAGE / "Pattapu.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-erza-pattapu-numerals.csv"
SHA256 = "065c351a516cf5350978c2b1c57ae50ea4affa0b6ee9085ee0414c294535ec77"
SOURCE = "erza2015pattapu"
NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1000, 2000]
NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen", "twenty"]
GLOSSES = {**dict(enumerate(NAMES, 1)),
           **{n: "twenty-" + NAMES[n - 21] for n in range(21, 30)},
           30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
           80: "eighty", 90: "ninety", 100: "hundred", 200: "two hundred",
           400: "four hundred", 800: "eight hundred", 1000: "thousand",
           2000: "two thousand"}


def source_units() -> list[dict]:
    raw = HTML.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Changed archived Pattapu HTML")
    tables = BeautifulSoup(raw, "html.parser").find_all("table")
    if len(tables) != 4 or len(tables[1].find_all("tr")) != 20:
        raise ValueError("Expected four HTML tables and twenty numeral rows")
    heading = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    credit = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    comment = re.sub(r"\s+", " ", tables[3].get_text(" ", strip=True))
    if "Pattapu" not in heading or "Erza" not in credit or "March 26" not in credit:
        raise ValueError("Changed language or contributor attribution")
    if "newly discovered Dravidian" not in comment:
        raise ValueError("Changed source comment")
    units = []
    for row_index, row in enumerate(tables[1].find_all("tr"), 1):
        cells = row.find_all("td")
        if len(cells) != 2:
            raise ValueError(f"Expected two cells in row {row_index}")
        for column, cell in enumerate(cells, 1):
            raw_cell = unicodedata.normalize("NFC", re.sub(r"\s+", " ", cell.get_text("", strip=False)).strip())
            pieces = [m for m in re.finditer(r"([0-9][0-9 ]*)\s*\.\s*([^,]+)", raw_cell)]
            if not pieces or ", ".join(m.group(0).strip() for m in pieces) != raw_cell:
                raise ValueError(f"Unparsed cell at {row_index}:{column}: {raw_cell!r}")
            for piece_index, match in enumerate(pieces, 1):
                number = int(match.group(1).replace(" ", ""))
                answers = [s.strip() for s in match.group(2).split("/")]
                if any(not s for s in answers):
                    raise ValueError(f"Empty answer at numeral {number}")
                units.append({"number": number, "printed_label": match.group(1),
                              "raw_cell": raw_cell, "raw_markup": str(cell),
                              "piece_index": piece_index, "table_row": row_index,
                              "table_column": column, "answers": answers})
    if len(units) != 42 or sorted(u["number"] for u in units) != NUMBERS:
        raise ValueError("Unexpected complete Pattapu prompt inventory")
    return sorted(units, key=lambda u: u["number"])


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        number = unit["number"]
        base = f"{SOURCE}:number:{number}"
        keys = []
        for answer_index, form in enumerate(unit["answers"], 1):
            key = f"{base}:answer:{answer_index}"
            keys.append(key)
            rows.append(["Pattapu", "", form, GLOSSES[number], "", form, "",
                         f"{SOURCE}[Pattapu table, numeral {number}]", "", "", key,
                         "", "", "", "num"])
        audit.append({**unit, "source_cell_key": base, "status": "ingested",
                      "language_id": "Pattapu", "source_lect": "Pattapu",
                      "gloss": GLOSSES[number], "entry_keys": keys,
                      "source_locator": f"HTML table 1, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "uncertainty": "source superscript ⁱ retained without phonological interpretation" if number in (5, 25) else "",
                      "review": "Slash in numeral 200 gives two complete alternatives, without a directional variant claim" if number == 200 else "source IPA-like transcription preserved in Original and Phonemic"})
    if len(rows) != 43:
        raise ValueError(f"Unexpected number of answers: {len(rows)}")
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
    print(f"{len(audit)} numbered items, {len(rows)} complete answers, zero holds")


if __name__ == "__main__":
    main()
