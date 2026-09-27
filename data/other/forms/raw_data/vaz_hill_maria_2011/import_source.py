"""Audit Vaz's archived Hill Madia numerals and the excluded Natarajan comparator."""

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
HTML = PACKAGE / "Maria-Abujh.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-vaz-hill-maria-numerals.csv"
SHA256 = "3655f54f967fa782c73ded31cf1fbc55ab9afa6789424cdb06bf4c9dc36221a7"
SOURCE = "vaz2011hillmaria"
DIALECT = "dialect:Maria%20%28India%29:vaz2011-hill-madia:Hill%20Madia%20%28Bhamragad%20area%29"
NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 200, 1000, 2000]
NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen", "twenty"]
GLOSSES = {**dict(enumerate(NAMES, 1)),
           **{n: "twenty-" + NAMES[n - 21] for n in range(21, 30)},
           30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
           80: "eighty", 90: "ninety", 100: "hundred", 200: "two hundred",
           1000: "thousand", 2000: "two thousand"}


def cell_text(cell) -> str:
    # Inline span boundaries are typography, not lexical spaces: 10 d̪ɘʔa,
    # 14 tʃɘvd̪a, 15 pɘnd̪ɾa, 17 sɘt̪ɾa, 18 ɘʈɾa.
    return re.sub(r"\s+", " ", cell.get_text("", strip=False)).strip()


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed archived Hill Maria HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 8 or any(len(tables[i].find_all("tr")) != 20 for i in (1, 5)):
        raise ValueError("Expected separate twenty-row Vaz and Natarajan numeral tables")
    heading = cell_text(tables[0])
    credit = cell_text(tables[2])
    second_heading = cell_text(tables[4])
    second_credit = cell_text(tables[6])
    if not all(("Hill Madia" in heading, "Christopher Vaz" in credit, "2011" in credit,
                "Abujhmaria" in second_heading, "Natarajan" in second_credit, "1985" in second_credit)):
        raise ValueError("Changed table boundaries or contributor attributions")
    units = []
    for table_name, table_index in (("Vaz 2011 Hill Madia", 1), ("Natarajan 1985 Abujhmaria", 5)):
        table_units = []
        for row_index, row in enumerate(tables[table_index].find_all("tr"), 1):
            cells = row.find_all("td")
            if len(cells) != 2:
                raise ValueError(f"Unexpected columns in table {table_index}, row {row_index}")
            for column, cell in enumerate(cells, 1):
                raw = cell_text(cell)
                match = re.fullmatch(r"([\d ]+)\s*\.\s*(.+)", raw)
                if not match:
                    raise ValueError(f"Unparsed numeral in table {table_index}: {raw!r}")
                number = int(match.group(1).replace(" ", ""))
                comments = re.findall(r"\([^()]*\)", match.group(2)) if table_index == 1 else []
                answer = re.sub(r"\s*\([^()]*\)", "", match.group(2)).strip() if table_index == 1 else match.group(2).strip()
                marker = "*" if answer.endswith("*") else ""
                if marker:
                    answer = answer[:-1].strip()
                table_units.append({"number": number, "printed_label": match.group(1),
                                    "source_cell_key": f"{SOURCE}:{'target' if table_index == 1 else 'comparator'}:number:{number}",
                                    "raw_cell": raw, "raw_markup": str(cell), "source_form": unicodedata.normalize("NFC", answer),
                                    "source_comment": comments, "unresolved_marker": marker,
                                    "table_name": table_name, "table_index": table_index,
                                    "table_row": row_index, "table_column": column})
        if len(table_units) != 40 or sorted(u["number"] for u in table_units) != NUMBERS:
            raise ValueError(f"Unexpected prompt inventory in {table_name}")
        units.extend(sorted(table_units, key=lambda item: item["number"]))
    return units


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        number = unit["number"]
        target = unit["table_index"] == 1
        key = unit["source_cell_key"] if target else ""
        if target:
            form = unit["source_form"]
            rows.append(["Maria (India)", "", form, GLOSSES[number], "", form, "",
                         f"{SOURCE}[Hill Madia table, numeral {number}]", "", "", key,
                         "", "", "", f"num {DIALECT}"])
        audit.append({**unit, "status": "ingested" if target else "excluded-comparator",
                      "reason": "" if target else "Separate Natarajan 1985 Abujhmaria grammar witness; not Vaz's elicitation",
                      "language_id": "Maria (India)" if target else "", "source_lect": "Hill Madia" if target else "Abujhmaria",
                      "gloss": GLOSSES[number], "entry_key": key,
                      "source_locator": f"HTML table {unit['table_index']}, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "review": ("Unexplained printed asterisk is retained as an audit marker; form itself is legible. " if unit["unresolved_marker"] else "") +
                                "No source-level arithmetic, borrowing or comparator relation is converted to a graph edge."})
    if len(rows) != 40 or len(audit) != 80:
        raise ValueError("Unexpected Hill Maria target/comparator counts")
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
    print(f"{len(audit)} audited cells, {len(rows)} target answers, 40 comparator cells excluded")


if __name__ == "__main__":
    main()
