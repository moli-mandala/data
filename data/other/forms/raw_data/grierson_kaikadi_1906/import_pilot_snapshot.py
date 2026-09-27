"""Import the complete numbered Kaikadi (Sholapur) column in LSI IV."""

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
INPUT = PACKAGE / "full-scope-inventory.tsv"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-grierson-kaikadi.csv"
SCAN = DATA.parent / "tmp/pdfs/LSI-V4.djvu"
SCAN_SHA256 = "33e9aaa220db22fcde712705581a69f7edcfc7e20b3e7a09347dc969768020f1"
SOURCE = "grierson1906lsi4"
DIALECT = "dialect:Kaikadi:kaikadi_sholapur_lsi1906:Sholapur"
INCLUDED = {"visually_reviewed", "existing_ingested"}
DEFERRED = {"deferred_transcription", "existing_deferred_transcription"}
BLANK = {"printed_blank", "existing_skipped_blank"}
EXCLUDED = {"excluded_sentence"}
ALL_STATUSES = INCLUDED | DEFERRED | BLANK | EXCLUDED


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [int(r["prompt"]) for r in rows] != list(range(1, 242)):
        raise ValueError("Expected precisely prompts 1–241")
    if any(not r["english_prompt"].strip() or not r["raw_cell"].strip() for r in rows):
        raise ValueError("Empty prompt gloss or source-cell description")
    for row in rows:
        prompt = int(row["prompt"])
        if row["status"] not in ALL_STATUSES:
            raise ValueError(f"Unreviewed or unknown status at prompt {prompt}")
        if (row["status"] in EXCLUDED) != (prompt >= 220):
            raise ValueError(f"Sentence scope/status mismatch at prompt {prompt}")
        if (row["status"] in INCLUDED) != bool(row["reviewed_form"]):
            raise ValueError(f"Status/form mismatch at prompt {row['prompt']}")
    counts = Counter(row["status"] for row in rows)
    if (sum(counts[s] for s in INCLUDED), sum(counts[s] for s in DEFERRED),
            sum(counts[s] for s in BLANK), sum(counts[s] for s in EXCLUDED)) != (162, 40, 17, 22):
        raise ValueError("Source-cell decisions changed; re-audit counts before install")
    return rows


def generate() -> tuple[list[list[str]], list[dict]]:
    installed: list[list[str]] = []
    audit: list[dict] = []
    for source_row in read_source():
        prompt = int(source_row["prompt"])
        page, djvu_page = int(source_row["printed_page"]), int(source_row["djvu_page"])
        if djvu_page != page + 20:
            raise ValueError(f"Page offset mismatch at prompt {prompt}")
        key = f"{SOURCE}:kaikadi_sholapur:{prompt}"
        forms = [unicodedata.normalize("NFC", form.strip())
                 for form in source_row["reviewed_form"].split(",") if form.strip()]
        entry_keys = []
        for index, form in enumerate(forms, start=1):
            entry_key = key if len(forms) == 1 else f"{key}:{index}"
            installed.append([
                "Kaikadi", "", form, source_row["english_prompt"], "", "", "",
                f"{SOURCE}[p. {page}, item {prompt}]", "", "", entry_key,
                "", "", "", DIALECT,
            ])
            entry_keys.append(entry_key)
        audit.append({
            "source_cell_key": key,
            "status": source_row["status"],
            "reason": source_row["reason"],
            "prompt": prompt,
            "english_prompt": source_row["english_prompt"],
            "language_id": "Kaikadi",
            "source_lect": "Kaikadi (Sholapur)",
            "dialect_tag": DIALECT,
            "printed_page": page,
            "djvu_page": djvu_page,
            "raw_cell": source_row["raw_cell"],
            "parsed_forms": forms,
            "entry_keys": entry_keys,
            "comparison_1928": "comparison-1928-full.tsv",
            "image_evidence": f"images/djvu-page-{djvu_page}-kaikadi-4x.png",
            "review": "visually checked against original scan column",
            "uncertainty": "transcription" if source_row["status"] in DEFERRED else "",
        })
    if len(audit) != 241 or len(installed) != 164:
        raise ValueError(f"Unexpected counts: {len(audit)} cells, {len(installed)} rows")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--check-scan", action="store_true")
    args = parser.parse_args()
    if args.check_scan and (not SCAN.exists() or hashlib.sha256(SCAN.read_bytes()).hexdigest() != SCAN_SHA256):
        raise SystemExit(f"Missing or changed source DjVu: {SCAN}")
    rows, audit = generate()
    print(f"{len(audit)} source cells, {len(rows)} installed rows")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
