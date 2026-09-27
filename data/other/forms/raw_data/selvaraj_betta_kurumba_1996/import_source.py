"""Audit Betta Kurumba source page; import Selvaraj's 1996 table only."""

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
HTML = PACKAGE / "Kurumba-Betta.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-selvaraj-betta-kurumba-numerals.csv"
SHA256 = "71b89e52ea2f514aa8051081ec8631772501f6f56c05ed25f0227250e39981c3"
SOURCE = "selvaraj1996bettakurumba"
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
        raise ValueError("Changed archived Betta Kurumba HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 10:
        raise ValueError("Expected ten HTML tables")
    first = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    second = re.sub(r"\s+", " ", tables[6].get_text(" ", strip=True))
    selvaraj = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    coelho = re.sub(r"\s+", " ", tables[8].get_text(" ", strip=True))
    if not ("Betta Kurumba" in first and "Betta Kurumba" in second
            and "Daniel Selvaraj" in selvaraj and "August 22, 1996" in selvaraj
            and "Gail Coelho" in coelho and "December 9, 2010" in coelho):
        raise ValueError("Changed source lect or contributor boundaries")
    units = []
    for table_index, scope in ((1, "target"), (7, "excluded_control")):
        rows = tables[table_index].find_all("tr")
        if len(rows) != 20:
            raise ValueError(f"Expected twenty rows in table {table_index}")
        for row_index, row in enumerate(rows, 1):
            cells = row.find_all("td")
            if len(cells) != 2:
                raise ValueError(f"Expected two cells at {table_index}:{row_index}")
            for column, cell in enumerate(cells, 1):
                joined = unicodedata.normalize("NFC", re.sub(r"\s+", " ", cell.get_text("", strip=False)).strip())
                match = re.match(r"([0-9][0-9 ]*)\s*\.\s*(.*)", joined)
                if not match or not match.group(2):
                    raise ValueError(f"Unparsed Betta Kurumba cell: {joined!r}")
                number = int(match.group(1).replace(" ", ""))
                control_body = match.group(2) if scope == "excluded_control" else ""
                layers = re.fullmatch(r"/([^/]+)/\s*(\[[^\]]+\])?", control_body) if control_body else None
                units.append({"scope": scope, "table_index": table_index, "table_row": row_index,
                              "table_column": column, "number": number, "printed_label": match.group(1),
                              "source_form": match.group(2) if scope == "target" else "",
                              "control_cell": joined if scope == "excluded_control" else "",
                              "control_phonemic": layers.group(1).strip() if layers else "",
                              "control_phonetic": layers.group(2).strip("[]") if layers and layers.group(2) else "",
                              "control_markup_status": ("well_delimited" if layers else "malformed_or_complex") if control_body else "",
                              "joined_cell": joined, "raw_markup": str(cell)})
    target = [u["number"] for u in units if u["scope"] == "target"]
    control = [u["number"] for u in units if u["scope"] == "excluded_control"]
    if len(units) != 80 or sorted(target) != NUMBERS or sorted(control) != NUMBERS:
        raise ValueError("Unexpected Betta Kurumba target/control inventory")
    return sorted(units, key=lambda u: (u["scope"] != "target", u["number"]))


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        target = unit["scope"] == "target"
        number = unit["number"]
        key = f"{SOURCE}:table:{unit['table_index']}:number:{number}"
        if target:
            form = unit["source_form"]
            rows.append(["BettaKurumba", "", form, GLOSSES[number], "", form, "",
                         f"{SOURCE}[Selvaraj Betta Kurumba table, numeral {number}]", "", "", key,
                         "", "", "", "num"])
        audit.append({**unit, "source_cell_key": key,
                      "status": "ingested" if target else "excluded_control",
                      "reason": "" if target else "Separately credited Coelho 2010/2011 phonemic and phonetic table; malformed delimiters and richer structure reserved for dedicated review.",
                      "language_id": "BettaKurumba", "source_lect": "Betta Kurumba",
                      "gloss": GLOSSES[number] if target else "", "entry_key": key if target else "",
                      "source_locator": f"HTML table {unit['table_index']}, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "uncertainty": "", "review": "No independence or revision relationship asserted between the two contributor tables."})
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
    print(f"{len(audit)} source cells audited: {len(rows)} selected 1996 readings, 40 excluded Coelho cells")


if __name__ == "__main__":
    main()
