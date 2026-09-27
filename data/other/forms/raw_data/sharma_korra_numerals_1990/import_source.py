"""Import Sharma's 1990 Korra Koraga numeral table from its HTML snapshot."""

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
HTML = PACKAGE / "Koraga-Korra.htm"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-sharma-korra-numerals.csv"
SHA256 = "a53cd1fc340e592bd3ebdcf53f8818ae3d22cc5b6c3020c329b7c3184b34a97c"
SOURCE = "sharma1990korra"
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
BLANK = set(range(21, 30)) | {200, 2000}


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed Korra Koraga HTML snapshot")
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
            match = re.fullmatch(r"([0-9 ]+)\s*\.\s*(.*)", raw)
            if not match:
                raise ValueError(f"Unparsed row {row_index}, column {column}: {raw}")
            label, form = match.groups()
            units.append({
                "number": int(label.replace(" ", "")),
                "raw_cell": raw,
                "printed_label": label,
                "source_form": unicodedata.normalize("NFC", form.strip()),
                "table_row": row_index,
                "table_column": column,
            })
    units.sort(key=lambda u: u["number"])
    if len(units) != 40 or [u["number"] for u in units] != sorted(GLOSSES):
        raise ValueError("Expected 40 numbered Korra Koraga cells")
    if {u["number"] for u in units if not u["source_form"]} != BLANK:
        raise ValueError("Changed Korra Koraga blank-cell inventory")
    return units


def generate() -> tuple[list[list[str]], list[dict]]:
    installed = []
    audit = []
    for unit in source_units():
        number = unit["number"]
        key = f"{SOURCE}:number:{number}"
        form = unit["source_form"]
        if form:
            installed.append([
                "Korra Koraga", "", form, GLOSSES[number], "", form, "",
                f"{SOURCE}[Korra Koraga table, numeral {number}]", "", "", key,
                "", "", "", "num",
            ])
        audit.append({
            **unit,
            "source_cell_key": key,
            "status": "ingested" if form else "source_blank",
            "reason": "" if form else "Numbered cell has no printed Korra Koraga reading.",
            "language_id": "Korra Koraga",
            "source_lect": "Korra Koraga",
            "gloss": GLOSSES[number],
            "parsed_form": form,
            "entry_key": key if form else "",
            "source_locator": f"Korra Koraga table, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
            "uncertainty": "",
            "review": "checked against source HTML; blank cells not inferred from compounds",
        })
    if len(audit) != 40 or len(installed) != 29:
        raise ValueError("Unexpected Korra Koraga counts")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit = generate()
    print(f"{len(audit)} numbered cells, {len(rows)} installed; {len(BLANK)} blank")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
