"""Import the Suketi column of Grierson 1916, LSI IX(IV), pp. 759–761.

Only prompts 1–13 and 32–79 are in scope. The TSV is a line-by-line manual
reading of the original-resolution public-domain scan; it records every cell
in that range, including printed ellipses and uncertain typography.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parents[4]
INPUT = PACKAGE / "transcription.tsv"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-grierson-suketi.csv"
PDF = DATA.parent / "tmp/pdfs/lsi-v9-4/LSI-V9-4.pdf"
PDF_SHA256 = "ef007663270b3a0ef5ba26804404e4db1281fb5eb239614cadf69b20b3d5395f"
SOURCE = "grierson1916suketi"
EXPECTED_PROMPTS = tuple([*range(1, 14), *range(32, 80)])
PDF_PAGE = {759: 775, 760: 776, 761: 777}  # one-based scan pages


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if tuple(int(r["prompt"]) for r in rows) != EXPECTED_PROMPTS:
        raise ValueError("Transcription must cover every target prompt 1–13 and 32–79")
    for row in rows:
        prompt = int(row["prompt"])
        expected_page = 759 if prompt <= 13 else 760 if prompt <= 52 else 761
        if int(row["printed_page"]) != expected_page:
            raise ValueError(f"Wrong printed page for prompt {prompt}")
        if not row["gloss"] or not row["suketi"]:
            raise ValueError(f"Missing gloss or source cell at prompt {prompt}")
        if row["decision"] not in {"ingest", "ambiguous", "blank", "skip_complex"}:
            raise ValueError(f"Unknown decision at prompt {prompt}")
        if row["decision"] == "blank" and row["suketi"] != "…":
            raise ValueError(f"Printed blank must use ellipsis at prompt {prompt}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        prompt = int(row["prompt"])
        page = int(row["printed_page"])
        locator = f"p. {page}, item {prompt}"
        cell_key = f"{SOURCE}:prompt:{prompt}"
        answers = [unicodedata.normalize("NFC", x.strip()) for x in row["suketi"].split(";")]
        if any(not x for x in answers):
            raise ValueError(f"Empty answer at prompt {prompt}")
        keys = []
        if row["decision"] == "ingest":
            for i, answer in enumerate(answers, 1):
                key = cell_key if i == 1 else f"{cell_key}:answer{i}"
                keys.append(key)
                installed.append([
                    "suk", "", answer, row["gloss"], "", "", "",
                    f"{SOURCE}[{locator}]", "", "", key, "", "", "", "",
                ])
        audit.append({
            "source_cell_key": cell_key,
            "status": "ingested" if row["decision"] == "ingest" else row["decision"],
            "reason": row["note"],
            "prompt": prompt,
            "english_prompt": row["gloss"],
            "printed_page": page,
            "printed_column": "Suketi",
            "pdf_page": PDF_PAGE[page],
            "raw_source_transcription": row["suketi"],
            "language_id": "suk",
            "source_lect": "Suketi",
            "dialect": "base lect; no more specific locality printed for table",
            "citation_locator": locator,
            "entry_keys": keys,
            "typography_review": (
                row["note"] if row["decision"] == "ambiguous"
                else "checked original-resolution scan; no unresolved italic, underline, or diacritic in installed spelling"
            ),
            "review": "manually checked against original LSI IX(IV) page image",
            "uncertainty": "transcription" if row["decision"] == "ambiguous" else "",
        })
    if len(audit) != 61 or len(installed) != 55:
        raise ValueError(f"Unexpected source counts: {len(audit)} cells, {len(installed)} rows")
    if len({r[10] for r in installed}) != len(installed):
        raise ValueError("Duplicate installed entry key")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--check-pdf", action="store_true")
    parser.add_argument("--audit-sample", action="store_true")
    args = parser.parse_args()
    if args.check_pdf:
        if not PDF.exists() or hashlib.sha256(PDF.read_bytes()).hexdigest() != PDF_SHA256:
            raise SystemExit(f"Missing or changed original-resolution scan: {PDF}")
    rows, audit = generate()
    print(f"{len(audit)} source cells, {len(rows)} installed rows, decisions={dict(Counter(a['status'] for a in audit))}")
    if args.audit_sample:
        for a in random.Random(1916).sample(audit, 20):
            print(f"{a['citation_locator']}: {a['raw_source_transcription']} -> {a['status']}")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
