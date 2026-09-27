"""Rebuild the visually reviewed Crooke 1892 Mirzapur Korwa glossary slice."""

import argparse
import csv
import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = "20260925-crooke-korwa-mirzapur"
SOURCE = "crooke1892korwa"
DIALECT = "dialect:kw:crooke-1892-mirzapur:Mirzapur%20Korwa"
EXPECTED_BY_PAGE = {125: 28, 126: 42, 127: 40, 128: 13}
EXPECTED_STATUSES = {"selected": 93, "held": 30}


def build():
    with (ROOT / "reviewed_inventory.tsv").open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    assert Counter(int(item["page"]) for item in inventory) == EXPECTED_BY_PAGE
    assert Counter(item["status"] for item in inventory) == EXPECTED_STATUSES
    seen = set()
    rows = []
    audit = []
    for item in inventory:
        page, number = int(item["page"]), int(item["item"])
        assert 1 <= number <= EXPECTED_BY_PAGE[page]
        key = f"{SOURCE}:mirzapur:p{page}:item{number:02}"
        assert key not in seen
        seen.add(key)
        form, gloss = item["form"].strip(), item["gloss"].strip()
        assert form and gloss
        assert form == unicodedata.normalize("NFC", form)
        note = item["note"].strip()
        status = item["status"]
        if status == "held":
            assert note
        audit.append({
            "entry_key": key,
            "printed_page": page,
            "scan_page": page + 10,
            "item": number,
            "gloss": gloss,
            "source_form": form,
            "status": status,
            "reason_or_note": note,
        })
        if status == "selected":
            rows.append([
                "kw", "", form, gloss, "", "", "",
                f"{SOURCE}[p. {page}, item {number}]",
                "", "", key, "", "", "", DIALECT,
            ])
    assert len(rows) == 93
    assert {x["item"] for x in audit if x["printed_page"] == 128} == set(range(1, 14))
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
    print(f"{len(rows)} Crooke Mirzapur Korwa forms prepared from {len(audit)} inventoried items; no database built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    write(parser.parse_args().install)
