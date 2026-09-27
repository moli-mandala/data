"""Import the complete Bailey 1920 Rohru glossary, pp. 127–130."""

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
OUTPUT = DATA / "data/other/forms/20260925-bailey-rohru.csv"
PDF = DATA.parent / "tmp/pdfs/bailey1920/bailey1920.pdf"
PDF_SHA256 = "7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39"
SOURCE = "bailey1920rohru"
SCOPE = [(127,"left",35),(127,"right",31),(128,"left",37),(128,"right",32),
         (129,"left",36),(129,"right",35),(130,"left",26),(130,"right",23)]


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [(int(r["page"]), r["column"], int(r["line"])) for r in rows] != [
        (page, column, i) for page, column, count in SCOPE for i in range(1, count + 1)
    ]:
        raise ValueError("Expected complete glossary pp. 127–130 (255 cells)")
    for row in rows:
        if not all(row[k] for k in ("gloss", "printed_form_review", "decision", "note")):
            raise ValueError(f"Incomplete source line {row['line']}")
        if row["decision"] not in {"ingest", "hold_typography", "exclude_crossref"}:
            raise ValueError(f"Invalid decision at line {row['line']}")
        if row["decision"] == "ingest" and "(?)" in row["printed_form_review"]:
            raise ValueError(f"Unresolved reading at line {row['line']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for row in read_source():
        line = int(row["line"])
        page, column = int(row["page"]), row["column"]
        key = f"{SOURCE}:p{page}:{column}:line:{line}"
        keys: list[str] = []
        if row["decision"] == "ingest":
            forms = [unicodedata.normalize("NFC", f.strip()) for f in row["printed_form_review"].split(";")]
            glosses = [g.strip() for g in row["gloss"].split(";")]
            if len(glosses) == 1:
                glosses *= len(forms)
            if len(forms) != len(glosses) or not all(forms + glosses):
                raise ValueError(f"Misaligned printed answers at line {line}")
            tags = row.get("tags", "").split(";")
            if len(tags) == 1:
                tags *= len(forms)
            if len(tags) != len(forms):
                raise ValueError(f"Misaligned tags at {key}")
            if (page, column, line) == (127, "left", 9):
                tags = ["noun"]
            for answer_no, (form, gloss, tag) in enumerate(zip(forms, glosses, tags), 1):
                entry_key = key if answer_no == 1 else f"{key}:answer{answer_no}"
                keys.append(entry_key)
                installed.append([
                    "roh", "", form, gloss, "", "", "",
                    f"{SOURCE}[p. {page}, {column} column, line {line}]",
                    "", "", entry_key, "", "", "", tag,
                ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested" if keys else row["decision"],
            "reason": row["note"],
            "printed_page": page,
            "scan_page": page + 26,
            "column": column,
            "editorial_line_number": line,
            "english_headword": row["gloss"],
            "printed_form_review": row["printed_form_review"],
            "language_id": "roh",
            "source_lect": "Rohru dialect of Koci",
            "source_line_ocr": row.get("source_line_ocr", ""),
            "citation_locator": f"p. {page}, {column} column, line {line}",
            "entry_keys": keys,
            "typography_review": "full page image checked; literal secure marks retained" if keys else row["note"],
            "uncertainty": "" if keys else row["decision"],
        })
    if len(audit) != 255:
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
    print(f"{len(audit)} units, {len(rows)} rows, decisions={dict(Counter(x['status'] for x in audit))}")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
