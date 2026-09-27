"""Import Bailey 1908 Kotguru lexical and numeral sections, printed pp. 30–33."""

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
LEXICAL_COMPLETION = PACKAGE / "completion-lexical.tsv"
NUMERAL_COMPLETION = PACKAGE / "completion-numerals.tsv"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-bailey-kotguru.csv"
PDF = DATA.parent / "tmp/pdfs/bailey-sainji/bailey1908.pdf"
PDF_SHA256 = "953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5"
SOURCE = "bailey1908kotguru"


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        row["column"] = "left"
        row["section"] = "lexical"
    with LEXICAL_COMPLETION.open(encoding="utf-8", newline="") as stream:
        lexical = list(csv.DictReader(stream, delimiter="\t"))
    for row in lexical:
        row["section"] = "lexical"
    with NUMERAL_COMPLETION.open(encoding="utf-8", newline="") as stream:
        numerals = list(csv.DictReader(stream, delimiter="\t"))
    rows = lexical + rows + numerals
    rows.sort(key=lambda r: (
        int(r["page"]),
        {"lexical": 0, "cardinal": 1, "ordinal": 2, "numeral_note": 3}[r["section"]],
        {"left": 0, "right": 1, "prose": 2}[r["column"]],
        int(r["item"]),
    ))
    expected = (
        [(30, "lexical", "left", i) for i in range(1, 21)]
        + [(30, "lexical", "right", i) for i in range(1, 23)]
        + [(31, "lexical", "left", i) for i in range(1, 43)]
        + [(31, "lexical", "right", i) for i in range(1, 43)]
        + [(32, "lexical", "left", i) for i in range(1, 14)]
        + [(32, "lexical", "right", i) for i in range(1, 13)]
        + [(32, "cardinal", "left", i) for i in range(1, 16)]
        + [(32, "cardinal", "right", i) for i in range(16, 30)]
        + [(32, "ordinal", "left", i) for i in range(1, 10)]
        + [(32, "ordinal", "right", i) for i in range(10, 18)]
        + [(33, "numeral_note", "prose", i) for i in range(1, 5)]
    )
    if [(int(r["page"]), r["section"], r["column"], int(r["item"])) for r in rows] != expected:
        raise ValueError("Expected the complete Kotguru lexical and numeral sections on pp. 30–33")
    for row in rows:
        if not row["gloss"] or not row["printed_form"]:
            raise ValueError(f"Blank source unit {row['item']}")
        if row["decision"] not in {"ingest", "hold_typography", "hold_incomplete", "hold_complex", "hold_morphology"}:
            raise ValueError(f"Invalid decision at {row['item']}")
        if row["decision"] == "ingest" and "(?)" in row["printed_form"]:
            raise ValueError(f"Unresolved form installed at {row['page']} {row['column']} {row['item']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        item = int(row["item"])
        page = int(row["page"])
        column = row["column"]
        section = row["section"]
        key = f"{SOURCE}:p{page}:{column}:item:{item}" if section == "lexical" else f"{SOURCE}:p{page}:{section}:{column}:item:{item}"
        locator = f"p. {page}, {column} column, item {item}" if section == "lexical" else (f"p. {page}, numeral prose note, example {item}" if section == "numeral_note" else f"p. {page}, {section}, {column} column, item {item}")
        keys: list[str] = []
        if row["decision"] == "ingest":
            forms = [unicodedata.normalize("NFC", x.strip()) for x in row["printed_form"].split(";")]
            if not all(forms):
                raise ValueError(f"Empty accepted answer at {item}")
            for answer_no, form in enumerate(forms, 1):
                entry_key = key if answer_no == 1 else f"{key}:answer{answer_no}"
                keys.append(entry_key)
                installed.append([
                    "Kotguru", "", form, row["gloss"], "", "", "",
                    f"{SOURCE}[{locator}]",
                    "", "", entry_key, "", "", "", "num" if section != "lexical" else "",
                ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested" if row["decision"] == "ingest" else row["decision"],
            "reason": row["note"],
            "printed_page": page,
            "scan_page": page + 22,
            "section": section,
            "column": column,
            "vocabulary_item": item,
            "english_headword": row["gloss"],
            "printed_form_review": row["printed_form"],
            "language_id": "Kotguru",
            "source_lect": "Kotguru",
            "citation_locator": locator,
            "entry_keys": keys,
            "typography_review": (
                "page image checked; printed length, underdots and hyphens retained"
                if row["decision"] == "ingest" else row["note"]
            ),
            "uncertainty": "" if row["decision"] == "ingest" else row["decision"],
        })
    if len(audit) != 201:
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
    print(f"{len(audit)} source units, {len(rows)} installed rows, decisions={dict(Counter(x['status'] for x in audit))}")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
