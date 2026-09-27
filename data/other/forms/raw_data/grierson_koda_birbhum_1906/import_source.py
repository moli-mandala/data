"""Reproduce the visually reviewed LSI IV Birbhum Koda prose-example pilot."""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = "20260925-grierson-koda-birbhum"
SOURCE = "grierson1906lsi4"
DIALECT = "dialect:Koda:koda_birbhum_lsi1906:Birbhum"


def build():
    audit = [json.loads(line) for line in (ROOT / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(audit) == 20
    assert Counter(item["printed_page"] for item in audit) == {109: 11, 110: 9}
    assert len({item["entry_key"] for item in audit}) == len(audit)
    assert Counter(item["status"] for item in audit) == {"ingested": 5, "held": 15}
    rows = []
    for item in audit:
        assert item["djvu_page"] == item["printed_page"] + 19
        assert item["entry_key"] == f"{SOURCE}:koda_birbhum:p{item['printed_page']}:ex{item['item']:02}"
        if item["status"] == "held":
            assert item["reason"]
            continue
        assert item["visual_reading"] and item["gloss"] and item["reason"] == ""
        rows.append([
            "Koda", "", item["visual_reading"], item["gloss"], "", "", "",
            f"{SOURCE}[p. {item['printed_page']}, Birbhum example {item['item']}]",
            "", "", item["entry_key"], "", "", "", DIALECT,
        ])
    assert len(rows) == 5
    return rows, audit


def write(install=False):
    rows, _ = build()
    path = ROOT / f"{STEM}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    if install:
        shutil.copyfile(path, ROOT.parents[1] / path.name)
    print(f"{len(rows)} Birbhum Koda forms prepared; no database built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    write(parser.parse_args().install)
