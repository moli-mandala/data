"""Import Bailey's complete 100-item Poguli vocabulary, printed pp. 58–59."""

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
OUTPUT = DATA / "data/other/forms/20260925-bailey-poguli.csv"
PDF = DATA.parent / "tmp/pdfs/bailey-sainji/bailey1908.pdf"
PDF_SHA256 = "953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5"
SOURCE = "bailey1908poguli"
LECT_TAG = "dialect:pog:bailey1908-poguli:Poguli%20%28Bailey%201908%29"


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [int(r["item"]) for r in rows] != list(range(1, 101)):
        raise ValueError("Expected all 100 consecutively numbered Poguli vocabulary items")
    for row in rows:
        item = int(row["item"])
        page, column = (58, "left") if item <= 36 else (58, "right") if item <= 71 else (59, "left") if item <= 86 else (59, "right")
        if (int(row["page"]), row["column"]) != (page, column):
            raise ValueError(f"Wrong page or column for item {item}")
        if not row["gloss"] or not row["note"]:
            raise ValueError(f"Incomplete item {item}")
        if row["decision"] not in {"ingest", "hold_typography", "hold_fragment", "hold_blank"}:
            raise ValueError(f"Invalid decision for item {item}")
        if not row["printed_form_review"] and row["decision"] != "hold_blank":
            raise ValueError(f"Missing reviewed text for item {item}")
        if row["decision"] == "ingest" and ("?" in row["printed_form_review"] or "-" in row["printed_form_review"]):
            raise ValueError(f"Unresolved/fragmentary installed form for item {item}")
    return rows


def tags_for(item: int) -> str:
    if item <= 13:
        return "num"
    if 59 <= item <= 67:
        return "verb"
    if 68 <= item <= 73:
        return "adv"
    if 74 <= item <= 76:
        return "pron"
    if 77 <= item <= 78:
        return "conj"
    if item == 82:
        return "interj"
    if item in {89, 90, 93, 94, 97}:
        return "noun pl"
    if item in {87, 88, 91, 95, 96}:
        return "noun sg"
    return ""


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        item = int(row["item"])
        page = int(row["page"])
        column = row["column"]
        source_key = f"{SOURCE}:p{page}:{column}:item:{item}"
        locator = f"p. {page}, {column} column, item {item}"
        entry_keys: list[str] = []
        if row["decision"] == "ingest":
            answers = row["printed_form_review"].split("|")
            for i, answer in enumerate(answers, 1):
                form = unicodedata.normalize("NFC", answer.strip())
                key = source_key if i == 1 else f"{source_key}:answer:{i}"
                entry_keys.append(key)
                installed.append([
                    "pog", "", form, row["gloss"], "", "", "",
                    f"{SOURCE}[{locator}]", "", "", key, "", "", "",
                    f"{LECT_TAG} {tags_for(item)}".strip(),
                ])
        audit.append({
            "source_cell_key": source_key,
            "status": "ingested" if entry_keys else row["decision"],
            "reason": row["note"],
            "printed_page": page,
            "scan_page": page + 224,
            "column": column,
            "vocabulary_item": item,
            "english_headword": row["gloss"],
            "printed_form_review": row["printed_form_review"],
            "language_id": "pog",
            "source_lect": "Poguli",
            "citation_locator": locator,
            "entry_keys": entry_keys,
            "review_state": "visually reviewed against original scan; no OCR installed",
            "uncertainty": "" if entry_keys else row["decision"],
        })
    expected = {"ingested": 89, "hold_typography": 4, "hold_fragment": 6, "hold_blank": 1}
    if len(audit) != 100 or len(installed) != 94 or dict(Counter(a["status"] for a in audit)) != expected:
        raise ValueError("Unexpected Poguli source/audit/installed counts")
    if len({r[10] for r in installed}) != len(installed):
        raise ValueError("Duplicate Poguli entry keys")
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
    print(f"{len(audit)} items, {len(rows)} rows, decisions={dict(Counter(a['status'] for a in audit))}")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
