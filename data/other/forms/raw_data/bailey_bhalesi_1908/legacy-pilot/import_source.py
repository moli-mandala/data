"""Import Bailey 1908 Bhalesi comparison list on printed pp. 73–74."""

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
OUTPUT = DATA / "data/other/forms/20260925-bailey-bhalesi.csv"
PDF = DATA.parent / "tmp/pdfs/bailey-sainji/bailey1908.pdf"
PDF_SHA256 = "953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5"
SOURCE = "bailey1908bhalesi"
DECISIONS = {"ingest", "hold_typography", "hold_complex", "hold_incomplete", "exclude_same_print"}


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [(int(r["page"]), r["column"], int(r["item"])) for r in rows] != (
        [(73, "left", i) for i in range(1, 7)]
        + [(73, "right", i) for i in range(7, 13)]
        + [(74, "left", i) for i in range(13, 24)]
        + [(74, "right", i) for i in range(24, 35)]
    ):
        raise ValueError("Expected all 34 Bhalesi p. 73–74 comparison-list lines")
    for row in rows:
        if not row["gloss"] or not row["printed_form_review"] or not row["note"]:
            raise ValueError(f"Incomplete item {row['item']}")
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Invalid decision at item {row['item']}")
        if row["decision"] == "ingest" and "(?)" in row["printed_form_review"]:
            raise ValueError(f"Unresolved form installed at item {row['item']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        item, page, column = int(row["item"]), int(row["page"]), row["column"]
        key = f"{SOURCE}:p{page}:{column}:item:{item}"
        keys: list[str] = []
        if row["decision"] == "ingest":
            forms = [unicodedata.normalize("NFC", x.strip()) for x in row["printed_form_review"].split(";")]
            if not all(forms):
                raise ValueError(f"Empty accepted answer at item {item}")
            for answer_no, form in enumerate(forms, 1):
                entry_key = key if answer_no == 1 else f"{key}:answer{answer_no}"
                keys.append(entry_key)
                installed.append([
                    "bhal", "", form, row["gloss"], "", "", "",
                    f"{SOURCE}[p. {page}, {column} column, item {item}]",
                    "", "", entry_key, "", "", "", "",
                ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested" if keys else row["decision"],
            "reason": row["note"],
            "printed_page": page,
            "scan_page": page + 114,
            "column": column,
            "vocabulary_item": item,
            "english_headword": row["gloss"],
            "printed_form_review": row["printed_form_review"],
            "language_id": "bhal",
            "source_lect": "Bhalesi",
            "citation_locator": f"p. {page}, {column} column, item {item}",
            "entry_keys": keys,
            "typography_review": "page image manually checked; clear macrons retained" if keys else row["note"],
            "uncertainty": "" if keys else row["decision"],
        })
    if len(audit) != 34 or len(installed) != 16:
        raise ValueError(f"Unexpected counts: {len(audit)} units, {len(installed)} rows")
    if len({r[10] for r in installed}) != len(installed):
        raise ValueError("Duplicate entry keys")
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
    print(f"{len(audit)} units, {len(rows)} rows, decisions={dict(Counter(x['status'] for x in audit))}")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
