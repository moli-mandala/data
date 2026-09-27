"""Import Bailey 1920 Bilaspuri's complete printed pp. 245–248 vocabulary."""

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
INPUT = PACKAGE / "full-transcription.tsv"
COMPLETION = PACKAGE / "completion.tsv"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-bailey-bilaspuri.csv"
PDF = DATA.parent / "tmp/pdfs/bailey1920/bailey1920.pdf"
PDF_SHA256 = "7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39"
SOURCE = "bailey1920bilaspuri"


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected = [(page, column, item)
                for page, column, count in ((245, "left", 33), (245, "right", 35),
                    (246, "left", 36), (246, "right", 37), (247, "left", 36),
                    (247, "right", 37), (248, "left", 13), (248, "right", 10))
                for item in range(1, count + 1)]
    if [(int(r["page"]), r["column"], int(r["item"])) for r in rows] != expected:
        raise ValueError("Expected every Bilaspuri glossary cell in printed column order")
    for row in rows:
        if not row["gloss"] or not row["printed_form"]:
            raise ValueError(f"Blank source unit {row['item']}")
        if row["decision"] not in {"ingest", "hold_typography", "hold_complex"}:
            raise ValueError(f"Invalid decision at {row['item']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        page = int(row["page"])
        column = row["column"]
        item = int(row["item"])
        key = f"{SOURCE}:p{page}:{column}:item:{item}"
        keys: list[str] = []
        if row["decision"] == "ingest":
            forms = [unicodedata.normalize("NFC", x.strip()) for x in row["printed_form"].split(";")]
            if not all(forms):
                raise ValueError(f"Empty accepted answer at {item}")
            glosses = [x.strip() for x in row["gloss"].split(";")]
            tags = row.get("tags", "").split(";")
            if len(glosses) == 1:
                glosses *= len(forms)
            if len(tags) == 1:
                tags *= len(forms)
            if len(glosses) != len(forms) or len(tags) != len(forms) or not all(glosses):
                raise ValueError(f"Misaligned senses/tags at {key}")
            for answer_no, (form, gloss, tag) in enumerate(zip(forms, glosses, tags), 1):
                entry_key = key if answer_no == 1 else f"{key}:answer{answer_no}"
                keys.append(entry_key)
                installed.append([
                    "bil", "", form, gloss, "", "", "",
                    f"{SOURCE}[p. {page}, {column} column, item {item}]",
                    "", "", entry_key, "", "", "", tag,
                ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested" if row["decision"] == "ingest" else row["decision"],
            "reason": row["note"],
            "printed_page": page,
            "scan_page": page + 26,
            "column": column,
            "vocabulary_item": item,
            "english_headword": row["gloss"],
            "printed_form_review": row["printed_form"],
            "language_id": "bil",
            "source_lect": "bil",
            "citation_locator": f"p. {page}, {column} column, item {item}",
            "entry_keys": keys,
            "source_line_ocr_aid": row.get("source_line_ocr", ""),
            "typography_review": (
                "page image checked; printed length, underdots and hyphens retained"
                if row["decision"] == "ingest" else row["note"]
            ),
            "uncertainty": "" if row["decision"] == "ingest" else row["decision"],
        })
    if len(audit) != 237 or len(installed) != 288:
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
