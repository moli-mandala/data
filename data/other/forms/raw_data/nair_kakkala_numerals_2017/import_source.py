"""Import Nair's 2017 Kakkala numeral table from its HTML snapshot."""

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
HTML = PACKAGE / "Kakkala.htm"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-nair-kakkala-numerals.csv"
SHA256 = "7e95db06a6c38b46d30b33ebb7fa921d457bf6518768739553425fa34de172b7"
SOURCE = "nair2017kakkala"
GLOSSES = {
    **{i: word for i, word in enumerate((
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty"
    ).split(), 1)},
    **{i: "twenty-" + word for i, word in enumerate((
        "one two three four five six seven eight nine"
    ).split(), 21)},
    30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    80: "eighty", 90: "ninety", 100: "hundred", 200: "two hundred",
    1000: "thousand", 2000: "two thousand",
}


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed Kakkala HTML snapshot")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 4:
        raise ValueError("Expected four page tables")
    rows = tables[1].find_all("tr")
    if len(rows) != 20:
        raise ValueError("Expected twenty numeral table rows")
    units = []
    for row_index, row in enumerate(rows, 1):
        cells = row.find_all("td")
        if len(cells) != 2:
            raise ValueError(f"Expected two cells at row {row_index}")
        for column, cell in enumerate(cells, 1):
            raw = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
            match = re.fullmatch(r"([0-9 ]+)\s*\.\s*(.+)", raw)
            if not match:
                raise ValueError(f"Unparsed row {row_index}, column {column}: {raw}")
            label, text = match.groups()
            number = int(label.replace(" ", ""))
            forms = [unicodedata.normalize("NFC", value.strip()) for value in text.split(" / ")]
            if not all(forms) or (len(forms) > 1 and number not in {10, 1000}):
                raise ValueError(f"Unexpected alternates at {number}: {text}")
            units.append({
                "number": number,
                "raw_cell": raw,
                "printed_label": label,
                "source_forms": forms,
                "table_row": row_index,
                "table_column": column,
            })
    units.sort(key=lambda u: u["number"])
    if len(units) != 40 or [u["number"] for u in units] != sorted(GLOSSES):
        raise ValueError("Expected 40 explicit numbered numeral prompts")
    if {u["number"]: len(u["source_forms"]) for u in units if len(u["source_forms"]) > 1} != {10: 2, 1000: 3}:
        raise ValueError("Changed Kakkala explicit alternatives")
    return units


def generate() -> tuple[list[list[str]], list[dict]]:
    installed = []
    audit = []
    for unit in source_units():
        number = unit["number"]
        entry_keys = []
        for reading, form in enumerate(unit["source_forms"], 1):
            key = f"{SOURCE}:number:{number}" + (f":reading:{reading}" if len(unit["source_forms"]) > 1 else "")
            entry_keys.append(key)
            installed.append([
                "Kakkala", "", form, GLOSSES[number], "", form, "",
                f"{SOURCE}[Kakkala table, numeral {number}]", "", "", key,
                "", "", "", "num",
            ])
        audit.append({
            **unit,
            "source_cell_key": f"{SOURCE}:number:{number}",
            "status": "ingested",
            "reason": "",
            "language_id": "Kakkala",
            "source_lect": "Kakkala (Kuḷupe:ccu)",
            "gloss": GLOSSES[number],
            "parsed_forms": unit["source_forms"],
            "entry_keys": entry_keys,
            "source_locator": f"Kakkala table, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
            "uncertainty": "",
            "review": "checked against source HTML; slash-separated complete answers retained separately; commentary's unprinted -ji/-cci alternants not generated",
        })
    if len(audit) != 40 or len(installed) != 43:
        raise ValueError("Unexpected Kakkala counts")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit = generate()
    print(f"{len(audit)} numbered items, {len(rows)} lexical readings")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
