"""Import the complete numbered Korvi (Belgaum) LSI IV column."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
INPUT = PACKAGE / "full-scope-inventory.tsv"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = PACKAGE.parents[4] / "data/other/forms/20260925-grierson-korvi.csv"
SCAN = PACKAGE.parents[5] / "tmp/pdfs/LSI-V4.djvu"
SCAN_SHA256 = "33e9aaa220db22fcde712705581a69f7edcfc7e20b3e7a09347dc969768020f1"
SOURCE = "grierson1906lsi4"
DIALECT = "dialect:Yerukula:korvi_belgaum_lsi1906:Belgaum"
PROMPTS = tuple(range(1, 242))
INCLUDED = {
    "visually_reviewed_candidate",
    "visually_reviewed_multiple_answers",
    "visually_reviewed_partial_multiple",
    "existing_ingested",
    "correction_pending_full_import",
    "correction_pending_full_import_uncertain",
}
DEFERRED = {
    "deferred_transcription",
    "existing_deferred_transcription",
}
EXCLUDED = {"excluded_full_sentence"}
ALL_STATUSES = INCLUDED | DEFERRED | EXCLUDED


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [int(row["prompt"]) for row in rows] != list(range(1, 242)):
        raise ValueError("Expected exactly prompts 1–241")
    for row in rows:
        prompt = int(row["prompt"])
        if row["status"] not in ALL_STATUSES:
            raise ValueError(f"Unreviewed or unknown status at prompt {prompt}")
        if (row["status"] in EXCLUDED) != (prompt >= 220):
            raise ValueError(f"Sentence scope/status mismatch at prompt {prompt}")
        if (row["status"] in INCLUDED) != bool(row["reviewed_form"].strip()):
            raise ValueError(f"Status/form mismatch at prompt {prompt}")
        if row["status"] == "visually_reviewed_partial_multiple" and prompt != 29:
            raise ValueError(f"Unexpected partial-answer cell at prompt {prompt}")
        if not row["english_prompt"].strip() or not row["raw_cell"].strip():
            raise ValueError(f"Missing printed-cell description at prompt {prompt}")
        if int(row["djvu_page"]) != int(row["printed_page"]) + 20:
            raise ValueError(f"Scan page offset mismatch at prompt {prompt}")
        if row["section"] == "sentence" and row["status"] not in EXCLUDED:
            raise ValueError(f"Sentence cell would be installed at prompt {prompt}")
        if not (PACKAGE / "images" / f"djvu-page-{row['djvu_page']}-korvi-4x.png").exists():
            raise ValueError(f"Missing original-scan image at prompt {prompt}")
    return rows


def generate() -> tuple[list[list[str]], list[dict]]:
    rows: list[list[str]] = []
    audit: list[dict] = []
    source_items = read_source()
    for item in source_items:
        prompt = int(item["prompt"])
        page = int(item["printed_page"])
        key = f"{SOURCE}:korvi_belgaum:{prompt}"
        forms = [unicodedata.normalize("NFC", part.strip())
                 for part in item["reviewed_form"].split(",") if part.strip()]
        tags = DIALECT
        if item["status"] == "correction_pending_full_import_uncertain":
            tags += " uncertain"
        entry_keys = []
        for index, form in enumerate(forms, start=1):
            if item["status"] == "visually_reviewed_partial_multiple":
                entry_key = f"{key}:2"
            elif len(forms) > 1:
                entry_key = f"{key}:{index}"
            else:
                entry_key = key
            rows.append([
                "Yerukula", "", form, item["english_prompt"], "", "", "",
                f"{SOURCE}[p. {page}, item {prompt}]", "", "", entry_key,
                "", "", "", tags,
            ])
            entry_keys.append(entry_key)
        reason = item["reason"]
        for old, new in (
            ("; not yet imported; compare source image before final install", "; checked against original source image"),
            ("; not yet imported", "; visually reviewed for this import"),
            ("; provisional pending final glyph check and importer", "; checked against original scan"),
            ("; correct on full-source regeneration", "; corrected in this full-source import"),
            ("Stage Ākḷ with source uncertainty when regenerating.", "Installed Ākḷ with source uncertainty."),
            ("Only second answer staged", "Only second answer installed"),
        ):
            reason = reason.replace(old, new)
        audit.append({
            "source_cell_key": key,
            "status": item["status"],
            "reason": reason,
            "prompt": prompt,
            "english_prompt": item["english_prompt"],
            "language_id": "Yerukula",
            "source_lect": "Korvi (Belgaum)",
            "dialect_tag": DIALECT,
            "tags": tags,
            "printed_page": page,
            "djvu_page": int(item["djvu_page"]),
            "raw_cell": item["raw_cell"],
            "parsed_forms": forms,
            "entry_keys": entry_keys,
            "comparison_1928": "comparison-1928-full.tsv",
            "image_evidence": f"images/djvu-page-{item['djvu_page']}-korvi-4x.png",
            "uncertainty": "transcription" if item["status"] in DEFERRED or item["status"] in {"correction_pending_full_import_uncertain", "visually_reviewed_partial_multiple"} else "",
        })
    if len(audit) != 241 or len(rows) != 156:
        raise ValueError("Unexpected full-source preview counts")
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--check-scan", action="store_true")
    args = parser.parse_args()
    if args.check_scan and (not SCAN.exists() or hashlib.sha256(SCAN.read_bytes()).hexdigest() != SCAN_SHA256):
        raise SystemExit(f"Missing or changed source DjVu: {SCAN}")
    rows, audit = generate()
    print(json.dumps({
        "printed_cells": len(audit),
        "form_rows": len(rows),
        "statuses": dict(Counter(item["status"] for item in audit)),
        "installed": args.install,
    }, ensure_ascii=False, indent=2))
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
