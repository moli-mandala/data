"""Import Bailey 1908 Kotkhai p. 24 lexical differences, all five cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parents[4]
INPUT = PACKAGE / "transcription.tsv"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-bailey-kotkhai.csv"
PDF = DATA.parent / "tmp/pdfs/bailey-sainji/bailey1908.pdf"
PDF_SHA256 = "953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5"
SOURCE = "bailey1908kotkhai"


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [(int(r["page"]), int(r["item"])) for r in rows] != [(24, i) for i in range(1, 6)]:
        raise ValueError("Expected every Kotkhai lexical difference on p. 24")
    for row in rows:
        if not row["gloss"] or not row["printed_form"]:
            raise ValueError(f"Blank source unit {row['item']}")
        if row["decision"] not in {"ingest", "hold_typography", "exclude_derivative"}:
            raise ValueError(f"Invalid decision at {row['item']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        item = int(row["item"])
        key = f"{SOURCE}:p24:item:{item}"
        keys: list[str] = []
        if row["decision"] == "ingest":
            form = unicodedata.normalize("NFC", row["printed_form"])
            keys = [key]
            installed.append([
                "Kotkhai", "", form, row["gloss"], "", "", "",
                f"{SOURCE}[p. 24, lexical difference {item}]",
                "", "", key, "", "", "", "",
            ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested" if row["decision"] == "ingest" else row["decision"],
            "reason": row["note"],
            "printed_page": 24,
            "scan_page": 46,
            "vocabulary_item": item,
            "english_headword": row["gloss"],
            "printed_form_review": row["printed_form"],
            "language_id": "Kotkhai",
            "source_lect": "Kotkhai",
            "citation_locator": f"p. 24, lexical difference {item}",
            "entry_keys": keys,
            "typography_review": "page image checked" if row["decision"] == "ingest" else row["note"],
            "uncertainty": "" if row["decision"] == "ingest" else row["decision"],
        })
    if len(audit) != 5 or len(installed) != 2:
        raise ValueError(f"Unexpected counts: {len(audit)} units, {len(installed)} rows")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--check-pdf", action="store_true")
    args = parser.parse_args()
    if args.check_pdf:
        digest = hashlib.sha256()
        with PDF.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != PDF_SHA256:
            raise SystemExit(f"Missing or changed original scan: {PDF}")
    rows, audit = generate()
    print(f"{len(audit)} source units, {len(rows)} installed rows, decisions={dict(Counter(x['status'] for x in audit))}")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
