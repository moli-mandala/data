"""Import the Baghi cells of Bailey's Rampur/Baghi complete vocabulary, pp. 144–147."""

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
INPUT = PACKAGE / "full-transcription.tsv"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-bailey-baghi.csv"
PDF = DATA.parent / "tmp/pdfs/bailey1920/bailey1920.pdf"
PDF_SHA256 = "7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39"
SOURCE = "bailey1920baghi"


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [(int(r["page"]), int(r["item"])) for r in rows] != [(p,i) for p,n in [(144,59),(145,66),(146,63),(147,57)] for i in range(1,n+1)]:
        raise ValueError("Expected all245 vocabulary units of printed pp144–147")
    for row in rows:
        if not row["gloss"] or not row["baghi_form"]:
            raise ValueError(f"Blank inventory at item {row['item']}")
        if row["decision"] not in {"ingest", "hold_typography", "hold_complex", "cross_reference", "hold_no_baghi"}:
            raise ValueError(f"Unknown decision at item {row['item']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        item = int(row["item"])
        page = int(row["page"])
        key = f"{SOURCE}:p{page}:item:{item}"
        keys: list[str] = []
        if row["decision"] == "ingest":
            forms = [unicodedata.normalize("NFC", s.strip()) for s in row["baghi_form"].split(";")]
            if not all(forms):
                raise ValueError(f"Empty Baghi answer at item {item}")
            glosses = row["gloss"].split(";")
            if len(glosses) == 1:
                glosses *= len(forms)
            tags = row.get("tags", "").split(";")
            if len(tags) == 1:
                tags *= len(forms)
            if len(glosses) != len(forms) or len(tags) != len(forms):
                raise ValueError(f"Gloss mismatch at {key}")
            for answer_no, (form, gloss, tag) in enumerate(zip(forms, glosses, tags), 1):
                entry_key = key if answer_no == 1 else f"{key}:answer{answer_no}"
                keys.append(entry_key)
                installed.append([
                    "ba", "", form, gloss, "", "", "",
                    f"{SOURCE}[p. {page}, vocabulary item {item}, Baghi column]",
                    "", "", entry_key, "", "", "", tag,
                ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested" if row["decision"] == "ingest" else row["decision"],
            "reason": row["note"],
            "printed_page": page,
            "scan_page": page + 26,
            "vocabulary_item": item,
            "column": row.get("column", ""),
            "column_cell": row.get("column_cell", ""),
            "raw_ocr_file": row.get("raw_ocr_file", ""),
            "english_headword": row["gloss"],
            "baghi_form_review": row["baghi_form"],
            "control_column": "Rampur: present before colon, excluded",
            "language_id": "ba",
            "source_lect": "Koci: Baghi dialect",
            "citation_locator": f"p. {page}, vocabulary item {item}, Baghi column",
            "entry_keys": keys,
            "typography_review": (
                "original page image checked; retained vowel lengths and consonant marks"
                if row["decision"] == "ingest" else row["note"]
            ),
            "uncertainty": "" if row["decision"] == "ingest" else row["decision"],
        })
    if len(audit) != 245 or len(installed) != 296:
        raise ValueError(f"Unexpected source counts: {len(audit)} units, {len(installed)} rows")
    if len({r[10] for r in installed}) != len(installed):
        raise ValueError("Duplicate entry keys")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--check-pdf", action="store_true")
    parser.add_argument("--audit-sample", action="store_true")
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
    if args.audit_sample:
        for item in random.Random(1921).sample(audit, 20):
            print(f"{item['citation_locator']}: {item['baghi_form_review']} -> {item['status']}")
    if args.install:
        with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
