"""Import Mini's 2018 Khirwar numeral table from its HTML snapshot."""

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
HTML = PACKAGE / "Khirwa.htm"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-mini-khirwar-numerals.csv"
SHA256 = "d6e7cfe2cee35dfddb843a0206ae294df3070083a327882eb7d7ed0f4de8cbd0"
SOURCE = "mini2018khirwar"
HELD = {
    5: "Source prints `p ã tʃɡo`, with unexplained internal spaces around nasal vowel; do not silently join into a lexical form.",
    50: "Source prints `p ã tʃasɡo`, with the same unexplained spacing; transcription held pending review.",
}
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
        raise ValueError("Changed Khirwar HTML snapshot")
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
            for part, piece in enumerate(re.split(r",\s*(?=(?:200|8 00)\.)", raw), 1):
                match = re.fullmatch(r"([0-9 ]+)\s*\.\s*(.+)", piece)
                if not match:
                    raise ValueError(f"Unparsed row {row_index}, column {column}: {piece}")
                label, form = match.groups()
                number = int(label.replace(" ", ""))
                units.append({
                    "number": number,
                    "raw_cell": raw,
                    "raw_piece": piece,
                    "printed_label": label,
                    "source_form": unicodedata.normalize("NFC", form.strip()),
                    "table_row": row_index,
                    "table_column": column,
                    "cell_part": part,
                })
    units.sort(key=lambda u: u["number"])
    if len(units) != 42 or [u["number"] for u in units] != sorted(GLOSSES):
        raise ValueError("Expected 42 explicit numbered numeral prompts")
    if {u["number"] for u in units if " " in u["source_form"]} != set(HELD):
        raise ValueError("Changed Khirwar source spacing")
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
                "Khirwar", "", form, GLOSSES[number], "", form, "",
                f"{SOURCE}[Khirwar table, numeral {number}]", "", "", key,
                "", "", "", "num loanword",
            ])
        audit.append({
            **unit,
            "source_cell_key": key,
            "status": status,
            "reason": HELD.get(number, ""),
            "language_id": "Khirwar",
            "source_lect": "Khirwar (Palamu)",
            "gloss": GLOSSES[number],
            "parsed_form": form,
            "entry_key": key if form else "",
            "source_locator": f"Khirwar table, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
            "uncertainty": "transcription" if status != "ingested" else "",
            "borrowing_claim": "The source states that Khirwar people have completely borrowed numerals from neighboring Indo-Aryan language; no specific donor or etymon identified.",
            "review": "checked against source HTML; no inferred component or donor links",
        })
    if len(audit) != 42 or len(installed) != 40:
        raise ValueError("Unexpected Khirwar counts")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit = generate()
    print(f"{len(audit)} numbered items, {len(rows)} installed; {len(HELD)} held")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
