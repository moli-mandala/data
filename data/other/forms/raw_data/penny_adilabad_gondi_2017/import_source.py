"""Import Penny's archived Western Southern/Adilabad Gondi numeral table."""

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
HTML = PACKAGE / "Gondi-Southern-Adilabad.htm"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-penny-adilabad-gondi-numerals.csv"
SHA256 = "11ad99a5fe0593cd2a8283ee8d596bb045b5266eef7c961c72f7033c2f8aef70"
SOURCE = "penny2017adilabadgondi"
NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 200, 1000, 2000]
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
    1000: "thousand", 2000: "two thousand",
}
QUALIFIER = re.compile(r"\s*(?:<\s*Indic|\(\s*native\s*\))\s*$", flags=re.I)


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed archived Adilabad Gondi HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 4 or len(tables[1].find_all("tr")) != 20:
        raise ValueError("Expected four HTML tables and twenty numeral rows")
    heading = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    credit = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    comment = re.sub(r"\s+", " ", tables[3].get_text(" ", strip=True))
    if ("Western Southern Gondi" not in heading or "Adilabad Gondi" not in heading
            or "Mark Penny" not in credit or "February 9, 2017" not in credit
            or "Adilabad Gondi" not in comment):
        raise ValueError("Changed source lect or contributor attribution")
    units = []
    for row_index, row in enumerate(tables[1].find_all("tr"), 1):
        cells = row.find_all("td")
        if len(cells) != 2:
            raise ValueError(f"Expected two cells in row {row_index}")
        for column, cell in enumerate(cells, 1):
            joined = unicodedata.normalize("NFC", re.sub(r"\s+", " ", cell.get_text("", strip=False)).strip())
            match = re.fullmatch(r"([0-9 ]+)\s*\.\s*(.+)", joined)
            if not match:
                raise ValueError(f"Unparsed numeral at {row_index}:{column}: {joined!r}")
            number = int(match.group(1).replace(" ", ""))
            raw_answers = re.split(r"\s*[/;,]\s*", match.group(2))
            if len(raw_answers) != (3 if number in {1000, 2000} else 2 if number in {9, 10, 11, 16, 20, 60} else 1):
                raise ValueError(f"Unexpected answer count for {number}: {raw_answers}")
            answers = []
            for raw_answer in raw_answers:
                qualifier = "Indic loan" if re.search(r"<\s*Indic\s*$", raw_answer, re.I) else (
                    "native" if re.search(r"\(\s*native\s*\)\s*$", raw_answer, re.I) else ""
                )
                form = QUALIFIER.sub("", raw_answer).strip()
                if not form or "<" in form or ")" in form:
                    raise ValueError(f"Unparsed qualifier for {number}: {raw_answer!r}")
                answers.append({"raw_answer": raw_answer, "source_form": form, "source_qualifier": qualifier})
            units.append({
                "number": number,
                "printed_label": match.group(1),
                "raw_cell": joined,
                "raw_markup": str(cell),
                "answers": answers,
                "table_row": row_index,
                "table_column": column,
            })
    if len(units) != 40 or sorted(u["number"] for u in units) != NUMBERS:
        raise ValueError("Unexpected Adilabad Gondi prompt inventory")
    return sorted(units, key=lambda x: x["number"])


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        number = unit["number"]
        key_base = f"{SOURCE}:number:{number}"
        entry_keys = []
        for answer_index, answer in enumerate(unit["answers"], 1):
            key = f"{key_base}:answer:{answer_index}"
            entry_keys.append(key)
            form = answer["source_form"]
            tags = "num loanword" if answer["source_qualifier"] == "Indic loan" else "num"
            rows.append([
                "Adilabad Gondi", "", form, GLOSSES[number], "", form, "",
                f"{SOURCE}[Western Southern Gondi table, numeral {number}]", "", "", key,
                "", "", "", tags,
            ])
        audit.append({
            **unit,
            "source_cell_key": key_base,
            "status": "ingested",
            "reason": "",
            "language_id": "Adilabad Gondi",
            "source_lect": "Western Southern Gondi or Adilabad Gondi",
            "gloss": GLOSSES[number],
            "entry_keys": entry_keys,
            "source_locator": f"HTML table 1, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
            "uncertainty": "",
            "review": "Only the explicitly qualified answer is tagged loanword; no donor or variant graph relation inferred.",
        })
    if len(rows) != 50 or sum("loanword" in r[14] for r in rows) != 8:
        raise ValueError("Unexpected Adilabad Gondi answer/loan count")
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit = generate()
    print(f"{len(audit)} numbered cells, {len(rows)} installed answers, 8 explicitly marked Indic loans")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
