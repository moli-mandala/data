"""Import Bailey's Sainji glossary and cardinal numerals (1908, pp. 55–56).

The checked-in TSV is a manual reading of the public-domain page images. It
accounts for every glossary line and numeral in this bounded scope. The PDF is
only needed for visual re-audit, never for deterministic CSV regeneration.
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
OUTPUT = DATA / "data/other/forms/20260925-bailey-sainji.csv"
PDF = DATA.parent / "tmp/pdfs/bailey-sainji/bailey1908.pdf"
PDF_SHA256 = "953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5"
SOURCE = "bailey1908sainji"
PDF_PAGE = {55: 77, 56: 78}  # one-based pages of the 358-page archive PDF
SKIPS = {"ambiguous", "skip_incomplete"}


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    glossary = [r for r in rows if r["section"] == "glossary"]
    numerals = [r for r in rows if r["section"] == "numerals"]
    if len(rows) != 66 or len(glossary) != 46 or len(numerals) != 20:
        raise ValueError("Expected exactly 46 glossary lines and 20 numerals")
    expected_glossary = {(str(col), str(item)) for col in (1, 2) for item in range(1, 24)}
    if {(r["column"], r["item"]) for r in glossary} != expected_glossary:
        raise ValueError("Glossary must contain rows 1–23 in each printed column")
    if {int(r["item"]) for r in numerals} != set(range(1, 21)):
        raise ValueError("Numerals must cover every printed number 1–20")
    if any(r["decision"] not in {"ingest", *SKIPS} for r in rows):
        raise ValueError("Unknown editorial decision")
    if any(not r["form"] or not r["gloss"] for r in rows):
        raise ValueError("Empty source form or gloss")
    keys = [(r["section"], r["page"], r["column"], r["item"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source-cell locator")
    return rows


def tags_for(row: dict[str, str]) -> str:
    if row["section"] == "numerals":
        return "num"
    # Bailey does not label the glossary entries by part of speech.
    return ""


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        page = int(row["page"])
        item = int(row["item"])
        col = int(row["column"])
        locator = (
            f"p. {page}, numeral {item}" if row["section"] == "numerals"
            else f"p. {page}, col. {col}, glossary row {item}"
        )
        cell_key = f"{SOURCE}:{row['section']}:{page}:{col}:{item}"
        variants = [unicodedata.normalize("NFC", x.strip()) for x in row["form"].split(";")]
        if any(not x for x in variants):
            raise ValueError(f"Empty variant in {cell_key}")
        entry_keys = []
        if row["decision"] == "ingest":
            for i, form in enumerate(variants, 1):
                key = cell_key if i == 1 else f"{cell_key}:answer{i}"
                entry_keys.append(key)
                installed.append([
                    "sai", "", form, row["gloss"], "", "", "",
                    f"{SOURCE}[{locator}]", "", "", key,
                    "", "", "", tags_for(row),
                ])
        audit.append({
            "source_cell_key": cell_key,
            "status": "ingested" if row["decision"] == "ingest" else row["decision"],
            "reason": row["note"],
            "section": row["section"],
            "item": item,
            "printed_page": page,
            "printed_column": col,
            "pdf_page": PDF_PAGE[page],
            "raw_source_transcription": row["form"],
            "raw_source_line": f"{row['form']}, {row['gloss']}",
            "gloss": row["gloss"],
            "language_id": "sai",
            "dialect": "Sainji (base lect; no more specific source site)",
            "citation_locator": locator,
            "entry_keys": entry_keys,
            "tags": tags_for(row),
            "review": "manually checked against Bailey 1908 page image",
            "typography_review": (
                row["note"] if row["decision"] == "ambiguous"
                else "checked; no unresolved underlined digraph or italic vowel in installed spelling"
            ),
            "uncertainty": "transcription" if row["decision"] == "ambiguous" else "",
        })
    if len(audit) != 66 or len(installed) != 55:
        raise ValueError(f"Unexpected counts: {len(audit)} source units, {len(installed)} rows")
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
            raise SystemExit(f"Missing or changed 358-page source PDF: {PDF}")
    installed, audit = generate()
    counts = Counter(str(a["status"]) for a in audit)
    print(f"{len(audit)} source units, {len(installed)} installed rows, decisions={dict(counts)}")
    if args.audit_sample:
        for row in random.Random(1908).sample(audit, 20):
            print(f"{row['citation_locator']}: {row['raw_source_line']} -> {row['status']}")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(installed)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
