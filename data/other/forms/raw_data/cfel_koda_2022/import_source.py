"""Reproduce the bounded, visually reviewed CFEL Koda print pilot; no DB build."""

import argparse
import csv
import json
import shutil
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = "20260925-cfel-koda-print-pilot"
SOURCE = "pradhan-tripathi2022koda"


def build():
    inventory = [json.loads(line) for line in (ROOT / "reviewed-inventory.jsonl").read_text().splitlines()]
    evidence = {item["api_id"]: item for item in
                (json.loads(line) for line in (ROOT / "api-evidence.jsonl").read_text().splitlines())}
    assert len(inventory) == 53 and len(evidence) == 17
    assert len({item["entry_key"] for item in inventory}) == 53
    rows = []
    for item in inventory:
        assert item["printed_page"] == item["pdf_page"] - 3
        if item["status"] != "ingested":
            assert item["status"] == "excluded" and item["reason"]
            continue
        source = evidence.pop(item["api_id"])
        assert source["query"] == item["english_head"]
        assert source["api_word"].strip() == item["printed_koda"]
        assert source["api_ipa"].strip() == item["api_ipa"]
        assert source["api_domain"] == item["api_domain"]
        assert source["response_sha256"] == item["api_response_sha256"]
        ipa = source["api_ipa"].strip()
        assert ipa.startswith("/") and ipa.endswith("/")
        ipa = ipa[1:-1]
        assert ipa and "?" not in ipa and "/" not in ipa
        rows.append([
            "Koda", "", unicodedata.normalize("NFC", ipa),
            item["english_head"], unicodedata.normalize("NFC", item["printed_koda"]),
            "", "", f"{SOURCE}[p. {item['printed_page']}, entry {item['item']}]",
            "", "", item["entry_key"], "", "", "", item["source_pos"],
        ])
    assert len(rows) == 17 and not evidence
    return rows, inventory


def write(install=False):
    rows, _ = build()
    path = ROOT / f"{STEM}.csv"
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    if install:
        shutil.copyfile(path, ROOT.parents[1] / path.name)
    print(f"{len(rows)} Koda rows prepared; no database built")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    write(parser.parse_args().install)
