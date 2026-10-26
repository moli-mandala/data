"""Import the complete Kāgānī vocabulary on Bailey 1920, printed pp. 106–109."""

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
OUTPUT = DATA / "data/other/forms/20260925-bailey-kagani.csv"
PDF = DATA.parent / "tmp/pdfs/bailey1920/bailey1920.pdf"
PDF_SHA256 = "7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39"
SOURCE = "bailey1920kagani"
DIALECT = "dialect:Northern%20Hindko:bailey1920-kagani:Kagani%20%28Bailey%201920%29"


def read_source(input_path: Path = INPUT) -> list[dict[str, str]]:
    with input_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    expected = [(p, c, i) for p, c, n in [(106,"left",32),(106,"right",35),(107,"left",35),(107,"right",33),(108,"left",35),(108,"right",35),(109,"left",27),(109,"right",29)] for i in range(1,n+1)]
    if [(int(r["page"]), r["column"], int(r["line"])) for r in rows] != expected:
        raise ValueError("Expected all 261 glossary cells, pp.106–109")
    for row in rows:
        if not row["gloss"] or not row["printed_form"]:
            raise ValueError(f"Blank inventory field at item {row['line']}")
        if row["decision"] not in {"ingest", "hold_typography", "hold_complex", "exclude_crossref"}:
            raise ValueError(f"Unknown decision at item {row['line']}")
    return rows


def generate(input_path: Path = INPUT) -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source(input_path):
        item = int(row["line"])
        page = int(row["page"])
        column = row["column"]
        key = f"{SOURCE}:p{page}:{column}:line:{item}"
        entry_keys: list[str] = []
        if row["decision"] == "ingest":
            answers = [unicodedata.normalize("NFC", s.strip()) for s in row["printed_form"].split(";")]
            if not all(answers):
                raise ValueError(f"Empty printed answer at item {item}")
            glosses = row["gloss"].split(";")
            tags = row.get("tags", "").split(";")
            if len(glosses) == 1:
                glosses *= len(answers)
            if len(tags) == 1:
                tags *= len(answers)
            if len(glosses) != len(answers) or len(tags) != len(answers):
                raise ValueError(f"Misaligned glosses or tags at {key}")
            for answer_no, (answer, gloss, tag) in enumerate(zip(answers, glosses, tags), 1):
                answer_key = key if answer_no == 1 else f"{key}:answer{answer_no}"
                entry_keys.append(answer_key)
                installed.append([
                    "Northern Hindko", "", answer, gloss, "", "", row.get("lexical_note", "") if answer_no == 1 else "",
                    f"{SOURCE}[p. {page}, {column} column, line {item}]", "", "", answer_key,
                    "", "", "", " ".join(filter(None, [DIALECT, tag])),
                ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested" if row["decision"] == "ingest" else row["decision"],
            "reason": row["note"],
            "printed_page": page,
            "scan_page": page + 26,
            "editorial_line_number": item,
            "column": column,
            "english_headword": row["gloss"],
            "printed_form_review": row["printed_form"],
            "source_lexical_note": row.get("lexical_note", ""),
            "tags_review": row.get("tags", ""),
            "source_line_ocr_aid": row.get("source_line_ocr", ""),
            "language_id": "Northern Hindko",
            "source_lect": "Kāgānī of the Kāgān Valley",
            "citation_locator": f"p. {page}, {column} column, line {item}",
            "entry_keys": entry_keys,
            "typography_review": (
                "original page image checked; printed vowel lengths and consonant marks preserved"
                if row["decision"] == "ingest" else row["note"]
            ),
            "uncertainty": row.get("uncertainty", "") if row["decision"] == "ingest" else row["decision"],
        })
    if len(audit) != 261:
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
        for item in random.Random(1920).sample(audit, 20):
            print(f"{item['citation_locator']}: {item['printed_form_review']} -> {item['status']}")
    if args.install:
        with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
