"""Audit and import the archived Riezen Eastern Muria numeral table."""

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
HTML = PACKAGE / "Muria-Eastern.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-riezen-eastern-muria-numerals.csv"
SHA256 = "2a7bdc4868c7c11d2fecc0cfe6f1ad69bfd9591c1f5683aeb1b74881cebbf558"
SOURCE = "riezen2013easternmuria"
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
        raise ValueError("Changed archived Eastern Muria HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 4 or len(tables[1].find_all("tr")) != 20:
        raise ValueError("Expected four tables and twenty numeral rows")
    heading = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    credit = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    if "Eastern Muria" not in heading or "Irene van Riezen" not in credit or "2013" not in credit:
        raise ValueError("Changed source lect or contributor attribution")
    units = []
    for row_index, row in enumerate(tables[1].find_all("tr"), 1):
        cells = row.find_all("td")
        if len(cells) != 2:
            raise ValueError(f"Expected two cells in row {row_index}")
        for column, cell in enumerate(cells, 1):
            raw = unicodedata.normalize("NFC", re.sub(r"\s+", " ", cell.get_text(" ", strip=True)))
            match = re.fullmatch(r"(\d+)\.\s*(.+)", raw)
            if not match:
                raise ValueError(f"Unparsed cell at {row_index}:{column}: {raw!r}")
            number = int(match.group(1))
            lexical = match.group(2)
            parentheticals = re.findall(r"\([^)]*\)", lexical)
            lexical = re.sub(r"\s*\([^)]*\)", "", lexical).strip()
            local_hindi = "as in Hindi" in lexical
            lexical = re.sub(r",?\s*as in Hindi\b", "", lexical).strip()
            status = "held" if number == 29 else "ingested"
            reason = "Unmatched closing bracket in printed form; lexical boundary unresolved" if status == "held" else ""
            units.append({"number": number, "source_cell_key": f"{SOURCE}:number:{number}",
                          "raw_cell": raw, "raw_markup": str(cell), "source_form": lexical,
                          "source_comment": parentheticals, "local_hindi": local_hindi,
                          "status": status, "reason": reason,
                          "table_row": row_index, "table_column": column})
    if len(units) != 40 or sorted(u["number"] for u in units) != NUMBERS:
        raise ValueError("Unexpected Eastern Muria prompt inventory")
    return sorted(units, key=lambda item: item["number"])


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        number = unit["number"]
        key = unit["source_cell_key"]
        form = unit["source_form"]
        if unit["status"] == "ingested":
            tags = "num loanword" if unit["local_hindi"] else "num"
            rows.append(["Eastern Muria", "", form, GLOSSES[number], "", form, "",
                         f"{SOURCE}[Eastern Muria table, numeral {number}]", "", "", key,
                         "", "", "", tags])
        audit.append({**unit, "language_id": "Eastern Muria", "source_lect": "Eastern Muria",
                      "gloss": GLOSSES[number], "entry_key": key if unit["status"] == "ingested" else "",
                      "source_locator": f"HTML table, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "review": "Only cell-local 'as in Hindi' is loan-tagged; source-wide contact prose is not a per-form claim."})
    if len(rows) != 39 or sum(a["status"] == "held" for a in audit) != 1:
        raise ValueError("Unexpected Eastern Muria accepted/held counts")
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
    print(f"{len(audit)} cells; {len(rows)} rows; one held bracketed cell")


if __name__ == "__main__":
    main()
