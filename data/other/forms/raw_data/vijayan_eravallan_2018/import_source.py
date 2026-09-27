"""Audit the two Eravallan tables; install only Vijayan's 2018 readings."""

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
HTML = PACKAGE / "Eravallan.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-vijayan-eravallan-numerals.csv"
SHA256 = "6d2510a53496c222055d81bd636629d79cdbac21b4e11932397cd8d23b69ab3d"
SOURCE = "vijayan2018eravallan"
NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 200, 400, 800, 1000, 2000]
CONTROL_NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 1000, 2000, 3000,
                                         4000, 5000, 6000, 7000, 8000, 9000, 10000]
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
        raise ValueError("Changed archived Eravallan HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 8:
        raise ValueError("Expected eight HTML tables")
    first = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    second = re.sub(r"\s+", " ", tables[4].get_text(" ", strip=True))
    credit_2018 = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    credit_2014 = re.sub(r"\s+", " ", tables[6].get_text(" ", strip=True))
    if not ("Eravallan" in first and "Eravallan" in second and "N. Vijayan" in credit_2018
            and "August 24, 2018" in credit_2018 and "V. Gnanasundaram" in credit_2014
            and "October 11, 2014" in credit_2014):
        raise ValueError("Changed contributor or lect boundaries")
    units = []
    for table_index, scope in ((1, "target"), (5, "excluded_control")):
        table_rows = tables[table_index].find_all("tr")
        if len(table_rows) != 20:
            raise ValueError(f"Expected twenty rows in table {table_index}")
        for row_index, row in enumerate(table_rows, 1):
            cells = row.find_all("td")
            if len(cells) != 2:
                raise ValueError(f"Expected two cells at {table_index}:{row_index}")
            for column, cell in enumerate(cells, 1):
                joined = unicodedata.normalize("NFC", re.sub(r"\s+", " ", cell.get_text("", strip=False)).strip())
                labels = list(LABEL.finditer(joined))
                if not labels:
                    raise ValueError(f"Unparsed Eravallan cell: {joined!r}")
                for segment_index, match in enumerate(labels, 1):
                    end = labels[segment_index].start() if segment_index < len(labels) else len(joined)
                    number = int(match.group(1).replace(" ", ""))
                    form = joined[match.end():end].strip(" ,; ")
                    if not form:
                        raise ValueError(f"Blank Eravallan reading at {table_index}:{number}")
                    units.append({"table_index": table_index, "scope": scope, "table_row": row_index,
                                  "table_column": column, "segment_index": segment_index,
                                  "joined_cell": joined, "raw_markup": str(cell), "printed_label": match.group(1),
                                  "number": number, "source_form": form})
    target = [u["number"] for u in units if u["scope"] == "target"]
    control = [u["number"] for u in units if u["scope"] == "excluded_control"]
    if len(units) != 89 or sorted(target) != NUMBERS or sorted(control) != CONTROL_NUMBERS:
        raise ValueError("Unexpected Eravallan target/control prompt inventory")
    return sorted(units, key=lambda u: (u["scope"] != "target", u["number"]))


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        target = unit["scope"] == "target"
        number = unit["number"]
        key = f"{SOURCE}:table:{unit['table_index']}:number:{number}"
        if target:
            form = unit["source_form"]
            rows.append(["Eravallan", "", form, GLOSSES[number], "", form, "",
                         f"{SOURCE}[Vijayan Eravallan table, numeral {number}]", "", "", key,
                         "", "", "", "num"])
        audit.append({**unit, "source_cell_key": key,
                      "status": "ingested" if target else "excluded_control",
                      "reason": "" if target else "Separately credited Gnanasundaram 2014 table; possibly related witness, outside selected 2018 table scope.",
                      "language_id": "Eravallan", "source_lect": "Eravallan",
                      "gloss": GLOSSES[number] if target else "", "entry_key": key if target else "",
                      "source_locator": f"HTML table {unit['table_index']}, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "uncertainty": "", "review": "No independent-fieldwork assertion is made for the two similar tables."})
    if len(rows) != 42:
        raise ValueError("Unexpected installed Eravallan count")
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
    print(f"{len(audit)} printed prompts audited: {len(rows)} selected 2018 readings, 47 excluded 2014 readings")


if __name__ == "__main__":
    main()
