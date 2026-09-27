"""Audit and import the archived Andronov/BSI Koya numeral table."""

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
HTML = PACKAGE / "Koya.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-andronov-koya-numerals.csv"
SHA256 = "0f9955a5cf981cedf29d2461fdd1a14d1268d4fe1d2833da13d5585648587803"
SOURCE = "andronov1995koya"
NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 200, 1000, 2000]
NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen", "twenty"]
GLOSSES = {**dict(enumerate(NAMES, 1)),
           **{n: "twenty-" + NAMES[n - 21] for n in range(21, 30)},
           30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
           80: "eighty", 90: "ninety", 100: "hundred", 200: "two hundred",
           1000: "thousand", 2000: "two thousand"}
PARTIAL = {22: {2}, 23: {2}, 200: {1}, 2000: {1}}


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed archived Koya HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 4 or len(tables[1].find_all("tr")) != 20:
        raise ValueError("Expected four HTML tables and twenty numeral rows")
    heading = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    credit = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    if "Koya" not in heading or "M. S. Andronov" not in credit or "November 23" not in credit:
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
            text = match.group(2)
            # Parentheticals are source commentary, never pieces of lexical forms.
            comments = re.findall(r"\([^)]*\)", text)
            text = re.sub(r"\s*\([^)]*\)", "", text).strip()
            segments = [s.strip() for s in re.split(r"\s*/\s*", text)]
            if number == 10:
                # Printed 'das ~ des' are two independently complete answers.
                segments = [segments[0], *[s.strip() for s in segments[1].split("~")]]
            if any(not s for s in segments):
                raise ValueError(f"Empty slash answer at {number}")
            answers = []
            for index, segment in enumerate(segments, 1):
                status = "held" if index in PARTIAL.get(number, set()) else "ingested"
                reason = ("printed slash is a compact constituent substitution; full answers cannot be recovered without expansion"
                          if status == "held" else "")
                answers.append({"answer_index": index, "source_form": segment,
                                "source_qualifier": "Indic loan" if "< Indic" in " ".join(comments) and (index == len(segments) or number == 10 and index == 2) else "",
                                "status": status, "reason": reason})
            units.append({"number": number, "printed_label": match.group(1), "raw_cell": joined,
                          "raw_markup": str(cell), "comments": comments, "answers": answers,
                          "table_row": row_index, "table_column": column})
    if len(units) != 40 or sorted(u["number"] for u in units) != NUMBERS:
        raise ValueError("Unexpected Koya prompt inventory")
    return sorted(units, key=lambda x: x["number"])


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        number = unit["number"]
        base = f"{SOURCE}:number:{number}"
        entry_keys = []
        for answer in unit["answers"]:
            if answer["status"] == "held":
                continue
            key = f"{base}:answer:{answer['answer_index']}"
            entry_keys.append(key)
            form = answer["source_form"]
            tags = "num loanword" if answer["source_qualifier"] == "Indic loan" else "num"
            rows.append(["Koya", "", form, GLOSSES[number], "", form, "",
                         f"{SOURCE}[Koya table, numeral {number}]", "", "", key,
                         "", "", "", tags])
        audit.append({**unit, "source_cell_key": base,
                      "status": "partial" if any(a["status"] == "held" for a in unit["answers"]) else "ingested",
                      "language_id": "Koya", "source_lect": "Koya", "gloss": GLOSSES[number],
                      "entry_keys": entry_keys,
                      "source_locator": f"HTML table 1, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "uncertainty": "printed component arithmetic says (2 x 20) beside hundred prompt" if number == 100 else "",
                      "review": "Slash and tilde answers are independent, not directional variants; only locally marked < Indic answers receive loanword tag."})
    if len(rows) != 58:
        raise ValueError(f"Unexpected Koya answer count: {len(rows)}")
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
    print(f"{len(audit)} numbered cells, {len(rows)} accepted answers, {sum(a['status']=='held' for u in audit for a in u['answers'])} held fragments")


if __name__ == "__main__":
    main()
