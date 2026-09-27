"""Import the credited Southeastern Kolami numeral table from the 2019 archive."""

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
HTML = PACKAGE / "Kolami-Southeastern.htm"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-kardile-southeastern-kolami-numerals.csv"
SHA256 = "883777607381ede03b7cca568aabd9661d61f35fd6173c503c77647e6f62e1e1"
SOURCE = "kardile2013southeasternkolami"
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


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed archived Southeastern Kolami HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 9:
        raise ValueError("Expected nine HTML tables, including two numeral tables")
    first_heading = tables[0].get_text(" ", strip=True)
    second_heading = tables[5].get_text(" ", strip=True)
    credits = tables[2].get_text(" ", strip=True)
    control_credits = tables[7].get_text(" ", strip=True)
    if not ("Southeastern Kolami" in first_heading and "Northwestern Kolami" in second_heading
            and "Subhangi" in credits and "March 13, 2013" in credits
            and "Malcolm Johnson" in control_credits):
        raise ValueError("Changed table labels or contributor attribution")
    units = []
    for table_index, status in ((1, "target"), (6, "excluded_control")):
        rows = tables[table_index].find_all("tr")
        if len(rows) != 20:
            raise ValueError(f"Expected twenty rows in numeral table {table_index}")
        seen = []
        for row_index, row in enumerate(rows, 1):
            cells = row.find_all("td")
            if len(cells) != 2:
                raise ValueError(f"Expected two cells at {table_index}:{row_index}")
            for column, cell in enumerate(cells, 1):
                raw = cell.get_text(" ", strip=True)
                normalized = unicodedata.normalize("NFC", re.sub(r"\s+", " ", raw).strip())
                match = re.fullmatch(r"([0-9 ]+)\s*\.\s*(.+)", normalized)
                if not match:
                    raise ValueError(f"Unparsed {table_index}:{row_index}:{column}: {raw!r}")
                printed_label, form = match.groups()
                number = int(printed_label.replace(" ", ""))
                seen.append(number)
                answers = [x.strip() for x in re.split(r"\s*/\s*", form)]
                if not answers or any(not x for x in answers):
                    raise ValueError(f"Empty slash answer: {normalized}")
                units.append({
                    "table_index": table_index,
                    "table_row": row_index,
                    "table_column": column,
                    "printed_label": printed_label,
                    "number": number,
                    "raw_cell": raw,
                    "source_form_cell": form,
                    "answers": answers,
                    "scope": status,
                })
        if sorted(seen) != NUMBERS:
            raise ValueError(f"Changed numeral inventory in table {table_index}")
    return units


def generate() -> tuple[list[list[str]], list[dict]]:
    rows = []
    audit = []
    for unit in source_units():
        target = unit["scope"] == "target"
        number = unit["number"]
        base_key = f"{SOURCE}:table:{unit['table_index']}:number:{number}"
        keys = []
        if target:
            for answer_index, form in enumerate(unit["answers"], 1):
                key = f"{base_key}:answer:{answer_index}"
                keys.append(key)
                rows.append([
                    "Naikri", "", form, GLOSSES[number], "", form, "",
                    f"{SOURCE}[Southeastern Kolami table, numeral {number}]", "", "",
                    key, "", "", "", "num",
                ])
        audit.append({
            **unit,
            "source_cell_key": base_key,
            "status": "ingested" if target else "excluded_control",
            "reason": "" if target else "Separately credited Northwestern Kolami comparator table; not part of target scope.",
            "language_id": "Naikri" if target else "Kolami",
            "source_lect": "Southeastern Kolami" if target else "Northwestern Kolami",
            "gloss": GLOSSES[number],
            "entry_keys": keys,
            "source_locator": f"HTML table {unit['table_index']}, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
            "uncertainty": "source_heading_conflict" if target else "",
            "review": (
                "English Southeastern heading and Kardile credit control mapping; archived Chinese heading and one prose sentence say Northwestern."
                if target else "Control table credited to Malcolm Johnson; not installed."
            ),
        })
    if len(audit) != 80 or len(rows) != 43 or len({r[10] for r in rows}) != 43:
        raise ValueError("Unexpected Southeastern Kolami counts")
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit = generate()
    print(f"{len(audit)} source cells, {len(rows)} target answers, 40 excluded control cells")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
