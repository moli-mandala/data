"""Import the Rāmpur cells of Bailey's paired Rampur/Baghi full vocabulary, pp. 144–147."""

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
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-bailey-rampur.csv"
PDF = DATA.parent / "tmp/pdfs/bailey1920/bailey1920.pdf"
PDF_SHA256 = "7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39"
SOURCE = "bailey1920rampur"


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [(int(r["page"]), int(r["item"])) for r in rows] != [(p,i) for p,n in [(144,59),(145,66),(146,63),(147,57)] for i in range(1,n+1)]:
        raise ValueError("Expected all245 vocabulary units pp144–147")
    for row in rows:
        if not row["gloss"] or not row["rampur_form_review"]:
            raise ValueError(f"Blank inventory at item {row['item']}")
        if row["decision"] not in {"ingest", "hold_typography", "hold_complex", "cross_reference"}:
            raise ValueError(f"Unknown decision at item {row['item']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        item = int(row["item"])
        page = int(row["page"])
        cell_key = f"{SOURCE}:p{page}:item:{item}"
        locator = f"p. {page}, vocabulary item {item}, Rampur column"
        keys: list[str] = []
        if row["decision"] == "ingest":
            forms = [unicodedata.normalize("NFC", s.strip()) for s in row["rampur_form_review"].split(";")]
            if not all(forms):
                raise ValueError(f"Empty accepted Rampur answer at item {item}")
            glosses = row["gloss"].split(";")
            tags = row.get("tags", "").split(";")
            if len(glosses) == 1:
                glosses *= len(forms)
            if len(tags) == 1:
                tags *= len(forms)
            if len(forms) != len(glosses) or len(forms) != len(tags):
                raise ValueError(f"Mismatched subanswers at {cell_key}")
            for answer_no, (form, gloss, tag) in enumerate(zip(forms, glosses, tags), 1):
                key = cell_key if answer_no == 1 else f"{cell_key}:answer{answer_no}"
                keys.append(key)
                installed.append([
                    "ramp", "", form, gloss, "", "", "",
                    f"{SOURCE}[{locator}]", "", "", key, "", "", "", tag,
                ])
        audit.append({
            "source_cell_key": cell_key,
            "status": "ingested" if keys else row["decision"],
            "reason": row["note"],
            "printed_page": page,
            "scan_page": page + 26,
            "vocabulary_item": item,
            "english_headword": row["gloss"],
            "visual_reading": row["rampur_form_review"],
            "reading_is_provisional": not bool(keys),
            "canonical_language": "ramp",
            "source_lect": "Kōci: Rāmpur dialect",
            "mapping_evidence": "Bailey 1920 p. 113 names the Rampur Koci dialect north of Rohru; p. 144 says answers before the colon belong to Rampur.",
            "control_column_excluded": "Bāghī: answers after colon; separately ingested as bailey1920baghi",
            "citation_locator": locator,
            "entry_keys": keys,
            "uncertainty": "" if keys else row["decision"],
        })
    if len(audit) != 245 or len(installed) != 288:
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
    print(f"{len(audit)} units, {len(rows)} installed rows, decisions={dict(Counter(a['status'] for a in audit))}")
    if args.install:
        with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
