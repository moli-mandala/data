"""Import Bailey 1908 Rambani numbered list, printed pp. 48–49, items 1–100."""

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
COMPLETION = PACKAGE / "completion.tsv"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-bailey-rambani.csv"
PDF = DATA.parent / "tmp/pdfs/bailey-sainji/bailey1908.pdf"
PDF_SHA256 = "953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5"
SOURCE = "bailey1908rambani"


def read_source() -> list[dict[str, str]]:
    rows = []
    for path in (INPUT, COMPLETION):
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream, delimiter="\t"))
    expected = (
        [(48, "left", i) for i in range(1, 37)]
        + [(48, "right", i) for i in range(37, 73)]
        + [(49, "left", i) for i in range(73, 87)]
        + [(49, "right", i) for i in range(87, 101)]
    )
    if [(int(r["page"]), r["column"], int(r["item"])) for r in rows] != expected:
        raise ValueError("Expected all 100 Rambani numbered items on pp. 48–49")
    for row in rows:
        if not row["gloss"] or not row["printed_form_review"] or not row["note"]:
            raise ValueError(f"Incomplete item {row['item']}")
        if row["decision"] not in {"ingest", "hold_typography", "hold_morphology", "exclude_same_print"}:
            raise ValueError(f"Invalid decision at item {row['item']}")
        if row["decision"] == "ingest" and "(?)" in row["printed_form_review"]:
            raise ValueError(f"Unresolved form installed at item {row['item']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        item = int(row["item"])
        page = int(row["page"])
        column = row["column"]
        key = f"{SOURCE}:p{page}:{column}:item:{item}"
        keys: list[str] = []
        if row["decision"] == "ingest":
            forms = [f.strip() for f in row["printed_form_review"].split(" | ")]
            for answer, form in enumerate(forms, 1):
                form = unicodedata.normalize("NFC", form)
                answer_key = key if len(forms) == 1 else f"{key}:answer:{answer}"
                keys.append(answer_key)
                installed.append([
                    "ram", "", form, row["gloss"], "", "", "",
                    f"{SOURCE}[p. {page}, {column} column, item {item}]",
                    "", "", answer_key, "", "", "", "num" if item <= 13 else "",
                ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested" if keys else row["decision"],
            "reason": row["note"],
            "printed_page": page,
            "scan_page": page + 224,
            "column": column,
            "vocabulary_item": item,
            "english_headword": row["gloss"],
            "printed_form_review": row["printed_form_review"],
            "language_id": "ram",
            "source_lect": "Rambani",
            "citation_locator": f"p. {page}, {column} column, item {item}",
            "entry_keys": keys,
            "typography_review": "page image manually checked; clear macrons retained" if keys else row["note"],
            "uncertainty": "" if keys else row["decision"],
        })
    if len(audit) != 100:
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
