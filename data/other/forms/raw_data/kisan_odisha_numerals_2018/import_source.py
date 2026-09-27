"""Import two credited 2018 Kisan numeral tables from the 2019 Chan snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata as ud
from pathlib import Path

from bs4 import BeautifulSoup


PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parents[4]
HTML = PACKAGE / "Kisan_Odisha.htm"
OUTPUT = DATA / "data/other/forms/20260925-kisan-odisha-numerals.csv"
AUDIT = PACKAGE / "audit.jsonl"
SHA256 = "a65613810ac0989604a473da70ca6b05fc1751947179008d33312abf0db969a6"
SOURCE = "kujur-perumalsamy2018kisan"
MARKER = re.compile(r"(?<!\d)(\d[\d ]*)\s*\.")
BRACKET = re.compile(r"^(.*?)\s*\[([^]]+)\]$")
BLANK_SECOND = set(range(21, 30)) | {800, 1000, 2000}
TYPOGRAPHY_REVIEW = {
    ("Kujur", 4), ("Kujur", 24), ("Kujur", 25), ("Kujur", 27),
    ("Kujur", 28), ("Kujur", 29), ("Perumalsamy", 4),
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
    400: "four hundred", 800: "eight hundred", 1000: "thousand", 2000: "two thousand",
}


def source_units() -> list[dict]:
    raw = HTML.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Changed Kisan archival HTML snapshot")
    tables = BeautifulSoup(raw, "html.parser").find_all("table")
    if len(tables) != 8:
        raise ValueError("Expected eight page tables")
    units = []
    for table_index, table_name in ((1, "Kujur"), (5, "Perumalsamy")):
        rows = tables[table_index].find_all("tr")
        if len(rows) != 20:
            raise ValueError(f"Expected 20 {table_name} table rows")
        for row_index, row in enumerate(rows, 1):
            cells = row.find_all("td")
            if len(cells) != 2:
                raise ValueError(f"Expected two cells in {table_name} row {row_index}")
            for column, cell in enumerate(cells, 1):
                raw_cell = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
                matches = list(MARKER.finditer(raw_cell))
                if not matches or matches[0].start() != 0:
                    raise ValueError(f"Unparsed {table_name} cell {row_index}/{column}: {raw_cell!r}")
                for item_index, match in enumerate(matches):
                    end = matches[item_index + 1].start() if item_index + 1 < len(matches) else len(raw_cell)
                    number = int(match.group(1).replace(" ", ""))
                    text = raw_cell[match.end():end].strip().strip(",").strip()
                    units.append({
                        "table": table_name,
                        "number": number,
                        "printed_label": match.group(1),
                        "raw_cell": raw_cell,
                        "raw_item": raw_cell[match.start():end].strip(),
                        "raw_answer": ud.normalize("NFC", text),
                        "table_row": row_index,
                        "table_column": column,
                        "item_in_cell": item_index + 1,
                    })
    if len(units) != 82:
        raise ValueError(f"Expected 82 numbered source units, found {len(units)}")
    for table in ("Kujur", "Perumalsamy"):
        selected = [u for u in units if u["table"] == table]
        if [u["number"] for u in selected if u["table_row"] == 5 and u["table_column"] == 1] != [4]:
            raise ValueError(f"Changed repeated printed 4 in {table} table")
    if {u["number"] for u in units if u["table"] == "Perumalsamy" and not u["raw_answer"]} != BLANK_SECOND:
        raise ValueError("Changed Perumalsamy blank-cell inventory")
    return units


def split_readings(answer: str) -> list[tuple[str, str]]:
    answer = re.sub(r"\s*<\s*Indo-Aryan\s*$", "", answer).strip()
    readings = []
    for part in answer.split(","):
        part = part.strip()
        if not part:
            continue
        match = BRACKET.fullmatch(part)
        readings.append((match.group(1).strip(), match.group(2).strip()) if match else (part, ""))
    return readings


def generate() -> tuple[list[list[str]], list[dict]]:
    installed, audit = [], []
    for unit in source_units():
        number, table = unit["number"], unit["table"]
        key = f"{SOURCE}:{table.lower()}:row:{unit['table_row']}:col:{unit['table_column']}:item:{unit['item_in_cell']}"
        repeated_four = number == 4 and unit["table_row"] == 5
        if repeated_four:
            status, reason = "held_ambiguous_prompt", "Second row printed 4; not silently relabelled as 5."
        elif not unit["raw_answer"]:
            status, reason = "source_blank", "Numbered cell has no printed reading."
        else:
            status, reason = "ingested", ""
        readings = split_readings(unit["raw_answer"]) if status == "ingested" else []
        output_keys = []
        for form_index, (form, ipa) in enumerate(readings, 1):
            form_key = key if len(readings) == 1 else f"{key}:answer:{form_index}"
            typography = (table, number) in TYPOGRAPHY_REVIEW
            tags = "num" + (" loanword" if number > 4 else "") + (" uncertain" if typography else "")
            installed.append([
                "Kurux", "", ud.normalize("NFC", form), GLOSSES[number], "",
                ud.normalize("NFC", ipa) if ipa else (form if table == "Kujur" else ""), "",
                f"{SOURCE}[{table} table, numeral {number}, row {unit['table_row']}]", "", "",
                form_key, "", "", "", f"{tags} dialect:Kurux:kisan-odisha:Kisan",
            ])
            output_keys.append(form_key)
        audit.append({
            **unit,
            "source_cell_key": key,
            "status": status,
            "reason": reason,
            "language_id": "Kurux",
            "dialect_tag": "dialect:Kurux:kisan-odisha:Kisan",
            "source_lect": "Kisan, Odisha",
            "gloss": GLOSSES.get(number, "ambiguous printed 4"),
            "parsed_readings": [[form, ipa] for form, ipa in readings],
            "entry_keys": output_keys,
            "source_locator": f"{table} table, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
            "uncertainty": "prompt numbering" if repeated_four else "transcription: fractured source spacing" if (table, number) in TYPOGRAPHY_REVIEW else "",
            "loan_note": "source claims numeral after 4 borrowed from Indo-Aryan, donor unspecified" if number > 4 and readings else "",
        })
    if len(audit) != 82:
        raise ValueError("Unexpected Kisan audit count")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit = generate()
    print(f"{len(audit)} numbered cells, {len(rows)} installed readings")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
