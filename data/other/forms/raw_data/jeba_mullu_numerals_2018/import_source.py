"""Import Jeba's 2018 Mullu Kurumba numeral list from Chan's HTML table."""

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
HTML = PACKAGE / "Kurumba-Mullu.htm"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-jeba-mullu-numerals.csv"
SHA256 = "12d2602bed82a9ac23832cdfa7f4e9085a8d266ed785cf63e935b9b9cbbede5a"
SOURCE = "jeba2018mullu"
HELD = {}
GLOSSES = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty", 21: "twenty-one", 22: "twenty-two", 23: "twenty-three",
    24: "twenty-four", 25: "twenty-five", 26: "twenty-six", 27: "twenty-seven",
    28: "twenty-eight", 29: "twenty-nine", 30: "thirty", 40: "forty",
    50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
    100: "hundred", 200: "two hundred", 400: "four hundred", 800: "eight hundred",
    1000: "thousand", 2000: "two thousand",
}


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed Mullu Kurumba HTML snapshot")
    soup = BeautifulSoup(data, "html.parser")
    tables = soup.find_all("table")
    if len(tables) != 4:
        raise ValueError("Expected four source-page tables")
    rows = tables[1].find_all("tr")
    if len(rows) != 20:
        raise ValueError("Expected twenty Mullu Kurumba numeral table rows")
    units = []
    for row_index, row in enumerate(rows, 1):
        cells = row.find_all("td")
        if len(cells) != 2:
            raise ValueError(f"Expected two cells at row {row_index}")
        for column, cell in enumerate(cells, 1):
            raw = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
            for part, piece in enumerate(re.split(r",\s*(?=(?:200|800)\.)", raw), 1):
                match = re.fullmatch(r"([0-9 ]+)\s*\.\s*(.+)", piece)
                if not match:
                    raise ValueError(f"Unparsed row {row_index}, column {column}: {piece}")
                label = match.group(1)
                number = int(label.replace(" ", ""))
                remainder = match.group(2)
                note = ""
                if " (" in remainder and remainder.endswith(")"):
                    remainder, note = remainder.rsplit(" (", 1)
                    note = "(" + note
                units.append({
                    "number": number,
                    "raw_cell": raw,
                    "raw_piece": piece,
                    "printed_label": label,
                    "source_form": unicodedata.normalize("NFC", remainder.strip()),
                    "arithmetic_note": note,
                    "table_row": row_index,
                    "table_column": column,
                    "cell_part": part,
                })
    units.sort(key=lambda r: r["number"])
    if len(units) != 42 or [u["number"] for u in units] != sorted(GLOSSES):
        raise ValueError("Expected 42 explicit numeral items and labels")
    return units


def generate() -> tuple[list[list[str]], list[dict]]:
    installed = []
    audit = []
    for unit in source_units():
        number = unit["number"]
        key = f"{SOURCE}:number:{number}"
        status = "deferred_transcription" if number in HELD else "ingested"
        form = unit["source_form"] if status == "ingested" else ""
        if form:
            installed.append([
                "Mullu Kurumba", "", form, GLOSSES[number], "", form, "",
                f"{SOURCE}[Mullu Kurumba table, numeral {number}]", "", "", key,
                "", "", "", "num",
            ])
        audit.append({
            **unit,
            "source_cell_key": key,
            "status": status,
            "reason": HELD.get(number, ""),
            "language_id": "Mullu Kurumba",
            "source_lect": "Mullu Kurumba",
            "gloss": GLOSSES[number],
            "parsed_form": form,
            "entry_key": key if form else "",
            "source_locator": f"Mullu Kurumba table, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
            "uncertainty": "transcription" if status != "ingested" else "",
            "review": "checked against source HTML table; no inferred component or cognacy links",
        })
    if len(audit) != 42 or len(installed) != 42:
        raise ValueError("Unexpected Mullu Kurumba counts")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit = generate()
    print(f"{len(audit)} explicit numerals, {len(rows)} installed rows")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
