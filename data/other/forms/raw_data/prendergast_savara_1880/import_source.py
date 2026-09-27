"""Build Prendergast's complete printed Savara vocabulary from reviewed cells."""

import argparse
import csv
import json
import re
import shutil
import unicodedata
from collections import Counter
from itertools import groupby
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEM = "20260925-prendergast-savara"
SOURCE = "prendergast1881savara"
DIALECT = "dialect:so:prendergast-1880-savara:Savara%20%28Prendergast%201880%29"
EXPECTED_CELLS = {(426, "left"): 59, (426, "right"): 56,
                  (427, "left"): 59, (427, "right"): 60,
                  (428, "left"): 23, (428, "right"): 9}


def build():
    inventory = []
    for filename in ("reviewed_inventory.tsv", "remaining_inventory.tsv"):
        with (ROOT / filename).open(encoding="utf-8", newline="") as handle:
            inventory.extend(csv.DictReader(handle, delimiter="\t"))
    assert len(inventory) == sum(EXPECTED_CELLS.values()) == 266
    assert Counter(item["status"] for item in inventory) == {"selected": 266}
    assert Counter((int(item["printed_page"]), item["column"]) for item in inventory) == EXPECTED_CELLS
    rows, audit = [], []
    for (page, column), group in groupby(inventory, key=lambda r: (int(r["printed_page"]), r["column"])):
      for index, item in enumerate(group, 1):
        assert int(item["item"]) == index
        assert item["english_prompt"] and item["printed_savara"] and item["note"]
        assert item["status"] == "selected"
        assert all(value == unicodedata.normalize("NFC", value) for value in item.values())
        key = f"{SOURCE}:p{page}:{column}:{index:02}"
        audit.append({
            "entry_key": key,
            "printed_page": page,
            "pdf_page": page + 46,
            "column": column,
            "item": index,
            "english_prompt": item["english_prompt"],
            "printed_savara": item["printed_savara"],
            "status": item["status"],
            "note": item["note"],
        })
        parts = [part.strip() for part in item["printed_savara"].split(", ")]
        assert len(parts) <= 2 and all(parts)
        for variant, part in enumerate(parts, 1):
            form = " ".join(re.sub(r"\([^)]*\)", "", part).split())
            child_key = key if variant == 1 else f"{key}:alt:{variant}"
            tags = DIALECT
            if "uncertain" in item["note"].lower():
                tags += " uncertain"
            rows.append([
                "so", "", form.casefold(), item["english_prompt"].casefold(),
                "", "", "",
                f"{SOURCE}[p. {page}, {column} column item {index}]", "", "", child_key,
                key if variant > 1 else "", "", "", tags,
            ])
    assert len(rows) == 270
    assert len({row[10] for row in rows}) == 270
    return rows, audit


def write(install=False):
    rows, audit = build()
    with (ROOT / f"{STEM}.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with (ROOT / "audit.jsonl").open("w", encoding="utf-8") as handle:
        for item in audit:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    if install:
        shutil.copyfile(ROOT / f"{STEM}.csv", ROOT.parents[1] / f"{STEM}.csv")
    print(f"{len(rows)} lexical facts from {len(audit)} printed cells; no database built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    write(parser.parse_args().install)
