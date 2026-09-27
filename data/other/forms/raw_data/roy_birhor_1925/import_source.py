"""Reproduce the complete-page Roy Birhor pilot without a data/DB build."""

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = "20260925-roy-birhor-p567-p568"
SOURCE = "roy1925birhors"


def build():
    audit = [json.loads(line) for line in (ROOT / "reviewed-inventory.jsonl").read_text().splitlines()]
    assert len(audit) == 59
    assert [(x["column"], x["column_item"]) for x in audit if x["printed_page"] == 567] == (
        [("L", n) for n in range(1, 10)] + [("R", n) for n in range(1, 15)]
    )
    assert [(x["column"], x["column_item"]) for x in audit if x["printed_page"] == 568] == (
        [("L", n) for n in range(1, 21)] + [("R", n) for n in range(1, 17)]
    )
    rows = []
    for item in audit:
        assert item["pdf_page"] == item["printed_page"] + 88
        assert item["printed_page"] in {567, 568}
        assert item["entry_key"] == f"roy1925birhor:p{item['printed_page']}:{item['column']}{item['column_item']:02}"
        if item["status"] != "ingested":
            assert item["status"] == "excluded" and item["reason"]
            continue
        assert "ch" not in item["printed_form"]
        note = f"Roy comparison: {item['source_comparison']}" if item["source_comparison"] else ""
        rows.append([
            "Birhor", "", item["printed_form"], item["gloss"], "", "", "",
            f"{SOURCE}[p. 567, {item['column']} column, item {item['column_item']}]",
            note, "", item["entry_key"], "", "", "", "",
        ])
    assert len(rows) == 46
    return rows, audit


def write(install=False):
    rows, _ = build()
    path = ROOT / f"{STEM}.csv"
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    if install:
        shutil.copyfile(path, ROOT.parents[1] / path.name)
    print(f"{len(rows)} Birhor rows prepared; no database built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    write(parser.parse_args().install)
