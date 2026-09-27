"""Import Jeba's 2018 Jennu Kurumba numeral table from the archived HTML."""

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
HTML = PACKAGE / "Jenu-Kurumba.htm"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-jeba-jennu-kurumba-numerals.csv"
SHA256 = "fa1685d1ba13da60ad6e4e5193b0304587a439b5c0e545ac055be44290867229"
SOURCE = "jeba2018jennukurumba"
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
        raise ValueError("Changed archived Jennu Kurumba HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 8:
        raise ValueError("Expected eight page tables")
    first = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    second = re.sub(r"\s+", " ", tables[4].get_text(" ", strip=True))
    jeba = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    control = re.sub(r"\s+", " ", tables[6].get_text(" ", strip=True))
    if not ("Jennu Kurumba" in first and "Jennu Kurumba" in second
            and "Melwin Jeba" in jeba and "September 8, 2018" in jeba
            and "Basavaraja Kodagunti" in control and "July 24, 2015" in control):
        raise ValueError("Changed lect or contributor boundaries")
    units = []
    target_seen = []
    for table_index, scope in ((1, "target"), (5, "excluded_control")):
        rows = tables[table_index].find_all("tr")
        if len(rows) != 20:
            raise ValueError(f"Expected twenty rows in table {table_index}")
        for row_index, row in enumerate(rows, 1):
            cells = row.find_all("td")
            if len(cells) != 2:
                raise ValueError(f"Expected two cells in {table_index}:{row_index}")
            for column, cell in enumerate(cells, 1):
                joined = unicodedata.normalize("NFC", re.sub(r"\s+", " ", cell.get_text("", strip=False)).strip())
                markup = str(cell)
                if scope == "excluded_control":
                    units.append({
                        "scope": scope, "table_index": table_index, "table_row": row_index,
                        "table_column": column, "control_cell": joined, "raw_markup": markup,
                        "number": None, "source_form": "", "printed_label": "", "segment_index": 0,
                    })
                    continue
                labels = list(LABEL.finditer(joined))
                expected = 2 if (row_index, column) in {(17, 2), (18, 2)} else 1
                if len(labels) != expected:
                    raise ValueError(f"Unexpected prompt labels at {row_index}:{column}: {joined!r}")
                for segment_index, match in enumerate(labels, 1):
                    end = labels[segment_index].start() if segment_index < len(labels) else len(joined)
                    form = joined[match.end():end].strip(" ,; ")
                    number = int(match.group(1).replace(" ", ""))
                    if not form:
                        raise ValueError(f"Blank Jennu reading at numeral {number}")
                    target_seen.append(number)
                    units.append({
                        "scope": scope, "table_index": table_index, "table_row": row_index,
                        "table_column": column, "control_cell": "", "raw_markup": markup,
                        "joined_cell": joined, "number": number, "source_form": form,
                        "printed_label": match.group(1), "segment_index": segment_index,
                    })
    if len(units) != 82 or sorted(target_seen) != NUMBERS:
        raise ValueError("Unexpected Jennu target/control inventory")
    return sorted(units, key=lambda x: (x["scope"] != "target", x["number"] or 0, x["table_row"], x["table_column"]))


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        target = unit["scope"] == "target"
        number = unit["number"]
        key = (f"{SOURCE}:table:1:number:{number}" if target else
               f"{SOURCE}:excluded-table:5:row:{unit['table_row']}:column:{unit['table_column']}")
        if target:
            form = unit["source_form"]
            rows.append([
                "Jennu Kurumba", "", form, GLOSSES[number], "", form, "",
                f"{SOURCE}[Jeba Jennu Kurumba table, numeral {number}]", "", "", key,
                "", "", "", "num",
            ])
        audit.append({
            **unit,
            "source_cell_key": key,
            "status": "ingested" if target else "excluded_control",
            "reason": "" if target else "Separately credited Kodagunti 2015 table in a different orthography, outside the Jeba table scope.",
            "language_id": "Jennu Kurumba",
            "source_lect": "Jennu Kurumba",
            "gloss": GLOSSES[number] if target else "",
            "entry_key": key if target else "",
            "source_locator": f"HTML table {unit['table_index']}, row {unit['table_row']}, column {unit['table_column']}, numeral {number}" if target else f"HTML table 5, row {unit['table_row']}, column {unit['table_column']}",
            "uncertainty": "",
            "review": "Adjacent inline spans joined without an invented space (notably numeral 26)." if target else "Comparator markup preserved; no values imported from second contributor.",
        })
    if len(rows) != 42 or len({r[10] for r in rows}) != 42:
        raise ValueError("Unexpected Jennu installed rows")
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit = generate()
    print(f"{len(audit)} audited units: {len(rows)} Jeba target prompts, 40 Kodagunti control cells")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
