"""Audit two Gadaba numeral tables; import John's Pottangi Ollar table only."""

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
HTML = PACKAGE / "Gadaba-Konekor.htm"
AUDIT = PACKAGE / "audit.jsonl"
DRAFT = PACKAGE / "draft.csv"
OUTPUT = PACKAGE.parents[1] / "20260925-john-pottangi-ollar-gadaba-numerals.csv"
SHA256 = "cea3111ad32229b6e564af62c187a43e1946d4b78182201d674505c82dc2ce63"
SOURCE = "john2013pottangiollargadaba"
NUMBERS = list(range(1, 31)) + [40, 50, 60, 70, 80, 90, 100, 200, 1000, 2000]
NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen", "twenty"]
GLOSSES = {**dict(enumerate(NAMES, 1)),
           **{n: "twenty-" + NAMES[n - 21] for n in range(21, 30)},
           30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
           80: "eighty", 90: "ninety", 100: "hundred", 200: "two hundred",
           1000: "thousand", 2000: "two thousand"}
QUALIFIER = re.compile(r"\s*(\([^)]*\))\s*$")


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed archived Ollar Gadaba HTML")
    tables = BeautifulSoup(data, "html.parser").find_all("table")
    if len(tables) != 10 or len(tables[1].find_all("tr")) != 20 or len(tables[7].find_all("tr")) != 20:
        raise ValueError("Expected two twenty-row tables in ten HTML tables")
    first = re.sub(r"\s+", " ", tables[0].get_text(" ", strip=True))
    second = re.sub(r"\s+", " ", tables[6].get_text(" ", strip=True))
    john = re.sub(r"\s+", " ", tables[2].get_text(" ", strip=True))
    sharma = re.sub(r"\s+", " ", tables[8].get_text(" ", strip=True))
    if not ("Pottangi Ollar Gadaba" in first and "Konekor Gadabar" in second
            and "Varghese John" in john and "March 6" in john and "2013" in john
            and "S. R. Sharma" in sharma and "December 3, 1990" in sharma):
        raise ValueError("Changed lect or contributor boundaries")
    units = []
    for table_index, scope in ((1, "target"), (7, "excluded_control")):
        for row_index, row in enumerate(tables[table_index].find_all("tr"), 1):
            cells = row.find_all("td")
            if len(cells) != 2:
                raise ValueError(f"Expected two cells at {table_index}:{row_index}")
            for column, cell in enumerate(cells, 1):
                joined = unicodedata.normalize("NFC", re.sub(r"\s+", " ", cell.get_text("", strip=False)).strip())
                match = re.fullmatch(r"([0-9][0-9 ]*)\s*\.\s*(.*)", joined)
                if not match:
                    raise ValueError(f"Unparsed Ollar Gadaba cell: {joined!r}")
                number = int(match.group(1).replace(" ", ""))
                body = match.group(2).strip()
                answers = []
                if scope == "target":
                    if not body:
                        raise ValueError(f"Blank target at numeral {number}")
                    segments = [s.strip() for s in body.split(",")] if number in {1, 2, 3} else [body]
                    if len(segments) != (3 if number in {1, 2, 3} else 1):
                        raise ValueError(f"Unexpected class answers at numeral {number}: {segments}")
                    for answer_index, segment in enumerate(segments, 1):
                        qualifier = QUALIFIER.search(segment)
                        source_qualifier = qualifier.group(1) if qualifier else ("< Oriya" if number == 4 else "")
                        form = QUALIFIER.sub("", segment).replace("< Oriya", "").strip()
                        if not form or "<" in form:
                            raise ValueError(f"Unparsed qualifier at {number}:{answer_index}: {segment!r}")
                        class_label = ("source prose: non-human" if answer_index == 1 and number in {1, 2, 3}
                                       else ("human masculine singular" if number == 1 else "human masculine") if answer_index == 2 and number in {1, 2, 3}
                                       else ("human feminine singular" if number == 1 else "human feminine") if answer_index == 3 and number in {1, 2, 3} else "")
                        tags = "num loanword" if number == 4 else (
                            "num m sg" if answer_index == 2 and number == 1 else
                            "num f sg" if answer_index == 3 and number == 1 else
                            "num m" if answer_index == 2 and number in {2, 3} else
                            "num f" if answer_index == 3 and number in {2, 3} else "num")
                        answers.append({"answer_index": answer_index, "source_form": form,
                                        "source_qualifier": source_qualifier, "source_class": class_label,
                                        "tags": tags, "raw_segment": segment})
                units.append({"scope": scope, "table_index": table_index, "table_row": row_index,
                              "table_column": column, "number": number, "printed_label": match.group(1),
                              "raw_cell": joined, "raw_markup": str(cell), "answers": answers,
                              "control_body": body if scope == "excluded_control" else "",
                              "control_blank": scope == "excluded_control" and not body})
    target = [u["number"] for u in units if u["scope"] == "target"]
    control = [u["number"] for u in units if u["scope"] == "excluded_control"]
    if len(units) != 80 or sorted(target) != NUMBERS or sorted(control) != NUMBERS:
        raise ValueError("Unexpected Ollar Gadaba target/control inventory")
    return sorted(units, key=lambda u: (u["scope"] != "target", u["number"]))


def generate() -> tuple[list[list[str]], list[dict]]:
    rows, audit = [], []
    for unit in source_units():
        target = unit["scope"] == "target"
        number = unit["number"]
        base = f"{SOURCE}:table:{unit['table_index']}:number:{number}"
        keys = []
        for answer in unit["answers"]:
            key = f"{base}:answer:{answer['answer_index']}"
            keys.append(key)
            form = answer["source_form"]
            rows.append(["OllariGadaba", "", form, GLOSSES[number], "", form, "",
                         f"{SOURCE}[Pottangi Ollar Gadaba table, numeral {number}]", "", "", key,
                         "", "", "", answer["tags"]])
        audit.append({**unit, "source_cell_key": base, "status": "ingested" if target else "excluded_control",
                      "reason": "" if target else "Separately credited Sharma 1990 Konekor Gadabar table, outside the John 2013 scope; blanks retained in audit.",
                      "language_id": "OllariGadaba", "source_lect": "Pottangi Ollar Gadaba" if target else "Konekor Gadabar",
                      "gloss": GLOSSES[number] if target else "", "entry_keys": keys,
                      "source_locator": f"HTML table {unit['table_index']}, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
                      "uncertainty": "first answer's non-human class inferred from local prose, not tagged as grammatical neuter" if target and number in {1, 2, 3} else "",
                      "review": "Only locally marked numeral 4 receives loanword; Oriya remains a source label, with no donor edge." if target and number == 4 else ""})
    if len(rows) != 46:
        raise ValueError("Unexpected Ollar Gadaba answer count")
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
    print(f"{len(audit)} source cells audited: {len(rows)} John answers, 40 excluded Sharma cells")


if __name__ == "__main__":
    main()
