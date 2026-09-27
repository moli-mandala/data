"""Audit the Sounderarajs' archived Bison-Horn Madiya numeral table."""

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
HTML = PACKAGE / "Maria-Dandami.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-sounderaraj-dandami-maria-numerals.csv"
SHA256 = "18eac514696ba851b9ec32b4f890463fe26f61cb9007ccb8555bdfd55c096eb2"
SOURCE = "sounderaraj1995dandamimaria"
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
        raise ValueError("Changed archived Dandami Maria HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 4 or len(tables[1].find_all("tr")) != 20:
        raise ValueError("Expected four tables and twenty numeral rows")
    heading = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    credit = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    if "Bison-Horn Madiya" not in heading or "Sounderaraj" not in credit or "1995" not in credit:
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
            answer = match.group(2).strip()
            comments = re.findall(r"\([^()]*\)", answer)
            answer = re.sub(r"\s*\([^()]*\)", "", answer).strip()
            forms = [unicodedata.normalize("NFC", fragment.strip()) for fragment in answer.split("/")]
            if any(not form for form in forms) or len(forms) > 2:
                raise ValueError(f"Unresolved alternatives at numeral {number}: {raw!r}")
            units.append({"number": number, "printed_label": match.group(1),
                          "source_cell_key": f"{SOURCE}:number:{number}",
                          "raw_cell": raw, "raw_markup": str(cell), "source_forms": forms,
                          "source_comment": comments, "table_row": row_index,
                          "table_column": column})
    if len(units) != 40 or sorted(u["number"] for u in units) != NUMBERS:
        raise ValueError("Unexpected Dandami prompt inventory")
    if {u["number"] for u in units if len(u["source_forms"]) == 2} != {20, 100}:
        raise ValueError("Changed printed answer alternatives")
    return sorted(units, key=lambda item: item["number"])


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        number = unit["number"]
        keys = []
        for answer_index, form in enumerate(unit["source_forms"], 1):
            key = f"{unit['source_cell_key']}:answer:{answer_index}"
            keys.append(key)
            rows.append(["Dandami Maria", "", form, GLOSSES[number], "", form, "",
                         f"{SOURCE}[Bison-Horn Madiya table, numeral {number}]", "", "", key,
                         "", "", "", "num"])
        audit.append({**unit, "status": "ingested", "reason": "", "language_id": "Dandami Maria",
                      "source_lect": "Bison-Horn Madiya", "gloss": GLOSSES[number],
                      "entry_keys": keys,
                      "source_locator": f"HTML table, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "review": "Slash readings are co-present source answers, not directional variants; arithmetic notes and broad borrowing prose do not establish form-level graph or donor claims."})
    if len(rows) != 42:
        raise ValueError("Unexpected Dandami answer count")
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
    print(f"{len(audit)} numbered cells, {len(rows)} source answers")


if __name__ == "__main__":
    main()
