"""Import the Simla Sirājī lexical column on LSI IX(IV), printed p. 631."""

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
OUTPUT = DATA / "data/other/forms/20260925-grierson-simla-siraji.csv"
PDF = DATA.parent / "tmp/pdfs/lsi-v9-4/LSI-V9-4.pdf"
PDF_SHA256 = "ef007663270b3a0ef5ba26804404e4db1281fb5eb239614cadf69b20b3d5395f"
SOURCE = "grierson1916simlasiraji"


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [(int(r["page"]), int(r["item"])) for r in rows] != [(631, i) for i in range(32, 53)]:
        raise ValueError("Expected all 21 Simla Sirājī target cells on printed p. 631")
    for row in rows:
        if not row["gloss"] or not row["printed_form_review"]:
            raise ValueError(f"Empty target cell {row['item']}")
        if row["decision"] not in {"ingest", "hold_typography"}:
            raise ValueError(f"Invalid decision at item {row['item']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for raw in read_source():
        item = int(raw["item"])
        locator = f"p. 631, standard-list item {item}"
        cell_key = f"{SOURCE}:p631:item:{item}"
        keys: list[str] = []
        if raw["decision"] == "ingest":
            answers = [unicodedata.normalize("NFC", x.strip()) for x in raw["printed_form_review"].split(";")]
            if not all(answers):
                raise ValueError(f"Empty accepted answer at item {item}")
            for answer_no, form in enumerate(answers, 1):
                key = cell_key if answer_no == 1 else f"{cell_key}:answer{answer_no}"
                keys.append(key)
                installed.append([
                    "ShimlaSiraji", "", form, raw["gloss"], "", "", "",
                    f"{SOURCE}[{locator}]", "", "", key, "", "", "", "",
                ])
        audit.append({
            "source_cell_key": cell_key,
            "status": "ingested" if keys else raw["decision"],
            "reason": raw["note"],
            "printed_page": 631,
            "scan_page": 647,
            "standard_list_item": item,
            "english_prompt": raw["gloss"],
            "visual_reading": raw["printed_form_review"],
            "reading_is_provisional": not bool(keys),
            "canonical_language": "ShimlaSiraji",
            "source_lect": "Simla Sirājī",
            "mapping_evidence": "Printed comparative-table column explicitly headed Simla Sirājī, matching the canonical Shimla Siraji label; no narrower site is given.",
            "control_column_excluded": "Śōrāchōlī",
            "citation_locator": locator,
            "entry_keys": keys,
            "uncertainty": "" if keys else raw["decision"],
        })
    if len(audit) != 21 or len(installed) != 16:
        raise ValueError(f"Unexpected counts: {len(audit)} cells, {len(installed)} rows")
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
    print(f"{len(audit)} cells, {len(rows)} installed rows, decisions={dict(Counter(a['status'] for a in audit))}")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
