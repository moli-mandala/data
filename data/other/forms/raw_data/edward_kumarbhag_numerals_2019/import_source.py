"""Import Edward's 2019 Kumarbhag Paharia numeral table from its HTML snapshot."""

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
HTML = PACKAGE / "Paharia-Kumarbhag.htm"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-edward-kumarbhag-numerals.csv"
SHA256 = "7272af1ba79fc7108d13d7b7deb9f0c7407094c85c2b1e5ad308ba11b01f7737"
SOURCE = "edward2019kumarbhag"
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
    400: "four hundred", 800: "eight hundred", 1000: "thousand",
    2000: "two thousand",
}


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed Kumarbhag Paharia HTML snapshot")
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
            pieces = re.split(r",\s*(?=(?:200|800)\.)", raw)
            for part, piece in enumerate(pieces, 1):
                match = re.fullmatch(r"([0-9 ]+)\.\s*(.+)", piece)
                if not match:
                    raise ValueError(f"Unparsed cell {row_index}/{column}: {piece}")
                label, content = match.groups()
                number = int(label.replace(" ", ""))
                notes = re.findall(r"\s*(\([^)]*\))", content)
                main = re.sub(r"\s*\([^)]*\)", "", content).strip()
                # The only slash separates two complete readings of 'twenty'.
                forms = [x.strip() for x in main.split(" / ")] if number == 20 else [main]
                if not all(forms) or (number == 20 and len(forms) != 2):
                    raise ValueError(f"Unexpected form split at {number}: {main}")
                units.append({
                    "number": number,
                    "raw_cell": raw,
                    "raw_piece": piece,
                    "printed_label": label,
                    "printed_parentheses": notes,
                    "source_forms": [unicodedata.normalize("NFC", x) for x in forms],
                    "table_row": row_index,
                    "table_column": column,
                    "cell_part": part,
                })
    units.sort(key=lambda u: u["number"])
    if len(units) != 42 or [u["number"] for u in units] != sorted(GLOSSES):
        raise ValueError("Expected 42 explicit numbered numeral prompts")
    return units


def generate() -> tuple[list[list[str]], list[dict]]:
    installed = []
    audit = []
    for unit in source_units():
        number = unit["number"]
        entry_keys = []
        for variant, form in enumerate(unit["source_forms"], 1):
            key = f"{SOURCE}:number:{number}" + (f":reading:{variant}" if len(unit["source_forms"]) > 1 else "")
            entry_keys.append(key)
            installed.append([
                "Kumarbhag Paharia", "", form, GLOSSES[number], "", form, "",
                f"{SOURCE}[Kumarbhag Paharia table, numeral {number}]", "", "", key,
                "", "", "", "num",
            ])
        audit.append({
            **unit,
            "source_cell_key": f"{SOURCE}:number:{number}",
            "status": "ingested",
            "reason": "",
            "language_id": "Kumarbhag Paharia",
            "source_lect": "Kumarbhag Paharia",
            "gloss": GLOSSES[number],
            "parsed_forms": unit["source_forms"],
            "entry_keys": entry_keys,
            "source_locator": f"Kumarbhag Paharia table, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
            "uncertainty": "",
            "review": "checked against source HTML; parenthetical transliterations/arithmetic kept in audit, not treated as lexical variants",
        })
    if len(audit) != 42 or len(installed) != 43:
        raise ValueError("Unexpected Kumarbhag Paharia counts")
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
