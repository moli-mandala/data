"""Import LSI V(I) 1903 Haijong (Mymensingh), printed p. 354 prompts 1–25."""

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
OUTPUT = DATA / "data/other/forms/20260925-grierson-haijong.csv"
SCAN = DATA.parent / "tmp/pdfs/lsi-v5-1/LSI-V5-1.djvu"
SCAN_SHA256 = "c1588bc4fb594caee445583402461e86c463a13fe3561dcdbc533a766ae0db67"
SOURCE = "grierson1903haijong"
DIALECT = "dialect:Hajong:lsi1903-haijong-mymensingh:Haijong%20%28Mymensingh%29"


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [int(r["prompt"]) for r in rows] != list(range(1, 26)):
        raise ValueError("Expected all 25 prompts in the first Haijong (Mymensingh) table page")
    for row in rows:
        if not all(row[k] for k in ("gloss", "printed_form_review", "decision", "note")):
            raise ValueError(f"Incomplete prompt {row['prompt']}")
        if row["decision"] not in {"ingest", "hold_typography"}:
            raise ValueError(f"Invalid decision for prompt {row['prompt']}")
        if row["decision"] == "ingest" and "(?)" in row["printed_form_review"]:
            raise ValueError(f"Unresolved reading installed for prompt {row['prompt']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        prompt = int(row["prompt"])
        key = f"{SOURCE}:p354:haijong:prompt:{prompt}"
        keys: list[str] = []
        if row["decision"] == "ingest":
            form = unicodedata.normalize("NFC", row["printed_form_review"])
            keys.append(key)
            installed.append([
                "Hajong", "", form, row["gloss"], "", "", "",
                f"{SOURCE}[p. 354, Haijong (Mymensingh) column, prompt {prompt}]",
                "", "", key, "", "", "", f"{DIALECT} num" if prompt <= 13 else DIALECT,
            ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested" if keys else "hold_typography",
            "reason": row["note"],
            "english_prompt_page": 352,
            "english_prompt_scan_page": 368,
            "target_printed_page": 354,
            "target_scan_page": 370,
            "target_column": "Haijong (Mymensingh), rightmost",
            "english_prompt_number": prompt,
            "english_gloss": row["gloss"],
            "printed_form_review": row["printed_form_review"],
            "language_id": "Hajong",
            "dialect_tag": DIALECT,
            "citation_locator": f"p. 354, Haijong (Mymensingh) column, prompt {prompt}",
            "entry_keys": keys,
            "uncertainty": "" if keys else "source typography",
            "control_columns": "English/Bengali on p. 352; Siripuri and Eastern Bengali on p. 354 excluded",
        })
    if len(audit) != 25 or len(installed) != 14:
        raise ValueError(f"Unexpected counts: {len(audit)} cells, {len(installed)} rows")
    if len({r[10] for r in installed}) != len(installed):
        raise ValueError("Duplicate entry keys")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--check-scan", action="store_true")
    args = parser.parse_args()
    if args.check_scan:
        digest = hashlib.sha256()
        with SCAN.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != SCAN_SHA256:
            raise SystemExit(f"Missing or changed original scan: {SCAN}")
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
