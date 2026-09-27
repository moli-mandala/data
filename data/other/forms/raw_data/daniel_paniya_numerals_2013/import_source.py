"""Import Daniel's 2013 Paniyan numeral table from its HTML snapshot."""

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
HTML = PACKAGE / "Paniyan.htm"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-daniel-paniya-numerals.csv"
SHA256 = "bc9b56163fe0be95b23e1c4cbff28894cf2ef1203f1503d70540ac9a51c36332"
SOURCE = "daniel2013paniya"
HELD = {
    8: "Source prints `ɘʈ;ʉ`; semicolon embedded in transcription has no explained function.",
    18: "Source prints `pɐjɪnɘʈː̪ʉ`; dental mark follows retroflex/length cluster and is unresolved.",
    23: "Source prints `ɪɾʉʋɐt̪ːɪmunːnːʉ`; duplicated length marks on n sequence are unresolved.",
    24: "Source prints `ɪɾʉʋɐt̪ːɪna̪lʉ`; dental mark appears on vowel a, likely typo but unverified.",
    28: "Source prints `ɪɾʉʋɐt̪ːɪjɘʈ;ʉ`; repeats unexplained semicolon from numeral 8.",
    70: "Source prints `ɘj̪ʉʋɐt̪ʉ`; dental mark appears on j, unresolved.",
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
    1000: "thousand", 2000: "two thousand",
}


def source_units() -> list[dict]:
    data = HTML.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Changed Paniyan HTML snapshot")
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
            match = re.fullmatch(r"([0-9 ]+)\s*\.\s*(.+)", raw)
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
        raise ValueError("Expected 40 explicit numbered numeral prompts")
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
                "Paniya", "", form, GLOSSES[number], "", form, "",
                f"{SOURCE}[Paniyan table, numeral {number}]", "", "", key,
                "", "", "", "num",
            ])
        audit.append({
            **unit,
            "source_cell_key": key,
            "status": status,
            "reason": HELD.get(number, ""),
            "language_id": "Paniya",
            "source_lect": "Paniyan",
            "language_mapping_note": "Page title/table/provider identify Paniyan = Paniya (pani1256); stray `Paliyan` in general commentary is not the table lect.",
            "gloss": GLOSSES[number],
            "parsed_form": form,
            "entry_key": key if form else "",
            "source_locator": f"Paniyan table, row {unit['table_row']}, column {unit['table_column']}, numeral {number}",
            "uncertainty": "transcription" if status != "ingested" else "",
            "review": "checked against source HTML; no inferred component or cognacy links",
        })
    if len(audit) != 40 or len(installed) != 34:
        raise ValueError("Unexpected Paniyan counts")
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
