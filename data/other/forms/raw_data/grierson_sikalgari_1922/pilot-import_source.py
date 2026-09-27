"""Install manually checked Sikalgari (Belgaum) cells from LSI XI, pp. 181–193."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parents[4]
INPUT = PACKAGE / "transcription.tsv"
AUDIT = PACKAGE / "audit.jsonl"
OUTPUT = DATA / "data/other/forms/20260925-grierson-sikalgari.csv"
PDF = DATA.parent / "tmp/pdfs/lsi-v11/LSI-V11.pdf"
PDF_SHA256 = "50df2b41e31420139148e2b16b321336881f227e917ca55cfedf225315cf4574"
SOURCE = "grierson1922lsi11"
PROMPTS = tuple([*range(1, 14), *range(32, 101)])
DIALECT = "dialect:Sik:sik_belgaum:Belgaum"


def page_for_prompt(prompt: int) -> tuple[int, int]:
    if prompt <= 25:
        return 181, 193
    if prompt <= 52:
        return 185, 197
    if prompt <= 79:
        return 189, 201
    return 193, 205


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if tuple(int(r["prompt"]) for r in rows) != PROMPTS:
        raise ValueError("Expected precisely prompts 1–13 and 32–100")
    if any(not r["gloss"].strip() or not r["form"].strip() for r in rows):
        raise ValueError("Empty gloss/form in manually checked source TSV")
    return rows


def generate() -> tuple[list[list[str]], list[dict]]:
    installed = []
    audit = []
    for source_row in read_source():
        prompt = int(source_row["prompt"])
        page, pdf_page = page_for_prompt(prompt)
        form = unicodedata.normalize("NFC", source_row["form"].strip())
        key = f"{SOURCE}:sikalgari_belgaum:{prompt}"
        installed.append([
            "Sik", "", form, source_row["gloss"], "", "", "",
            f"{SOURCE}[p. {page}, item {prompt}]", "", "", key,
            "", "", "", DIALECT,
        ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested",
            "reason": "",
            "prompt": prompt,
            "english_prompt": source_row["gloss"],
            "language_id": "Sik",
            "source_lect": "Sikalgari (Belgaum)",
            "dialect_tag": DIALECT,
            "printed_page": page,
            "pdf_page": pdf_page,
            "raw_cell": form,
            "entry_key": key,
            "ocr_comparison": f"ocr/pdf-page-{pdf_page}-sikalgari.txt",
            "review": "manually checked against original page image",
            "uncertainty": "",
        })
    if len(audit) != 82 or len(installed) != 82:
        raise ValueError(f"Unexpected source counts: {len(audit)} cells, {len(installed)} rows")
    return installed, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--check-pdf", action="store_true")
    args = parser.parse_args()
    if args.check_pdf and (not PDF.exists() or hashlib.sha256(PDF.read_bytes()).hexdigest() != PDF_SHA256):
        raise SystemExit(f"Missing or changed source PDF: {PDF}")
    rows, audit = generate()
    print(f"{len(audit)} source cells, {len(rows)} installed rows")
    if args.install:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
