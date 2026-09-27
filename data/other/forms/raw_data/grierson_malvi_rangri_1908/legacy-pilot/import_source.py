"""Import bounded LSI IX(II) Mālvī (Rāngrī) comparative-table cells."""

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
OUTPUT = DATA / "data/other/forms/20260925-grierson-malvi-rangri.csv"
SCAN = DATA.parent / "tmp/pdfs/lsi-v9-2/LSI-V9-2.djvu"
SCAN_SHA256 = "d6796bad8d267b776d2aec40dabb0b3b74d4f4fc87eb118f81db1b88e9dc6049"
SOURCE = "grierson1908malvirangri"
DIALECT = "dialect:Malw:lsi1908-malvi-rangri:Rangri"


def read_source() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [(int(r["page"]), int(r["item"])) for r in rows] != [(307, i) for i in range(32, 53)]:
        raise ValueError("Expected exactly Mālvī (Rāngrī) p. 307 prompts 32–52")
    for row in rows:
        if not row["gloss"] or not row["printed_form_review"] or not row["standard_malvi_control"]:
            raise ValueError(f"Incomplete source cell {row['item']}")
        if row["decision"] not in {"ingest", "hold_typography"}:
            raise ValueError(f"Unknown decision for source cell {row['item']}")
    return rows


def generate() -> tuple[list[list[str]], list[dict[str, object]]]:
    installed: list[list[str]] = []
    audit: list[dict[str, object]] = []
    for raw in read_source():
        item = int(raw["item"])
        key = f"{SOURCE}:p307:rangri:item:{item}"
        locator = f"p. 307, Mālvī (Rāngrī) column, standard-list item {item}"
        accepted = raw["decision"] == "ingest"
        form = unicodedata.normalize("NFC", raw["printed_form_review"])
        if accepted:
            if "?" in form or ";" in form:
                raise ValueError(f"Unresolved or multiple accepted form at item {item}")
            installed.append([
                "Malw", "", form, raw["gloss"], "", "", "",
                f"{SOURCE}[{locator}]", "", "", key, "", "", "", DIALECT,
            ])
        audit.append({
            "source_cell_key": key,
            "status": "ingested" if accepted else raw["decision"],
            "reason": raw["note"],
            "printed_page": 307,
            "scan_page": 322,
            "standard_list_item": item,
            "english_prompt": raw["gloss"],
            "rangri_visual_reading": form,
            "reading_is_provisional": not accepted,
            "standard_malvi_control": raw["standard_malvi_control"],
            "control_excluded": True,
            "canonical_language": "Malw",
            "source_lect": "Mālvī (Rāngrī)",
            "dialect_tag": DIALECT,
            "mapping_evidence": "LSI IX(II) p. 52 calls Rāngrī a form of Mālvī spoken by Rajputs of Malwa proper; p. 240 treats Standard Mālvī and Rāngrī specimens; Glottolog malv1243 is Malvi, matching Jambu Malw despite legacy display name Malwai.",
            "citation_locator": locator,
            "entry_keys": [key] if accepted else [],
            "uncertainty": "" if accepted else "transcription:printed-glyph",
        })
    if len(audit) != 21 or len(installed) != 5:
        raise ValueError(f"Unexpected counts: {len(audit)} cells, {len(installed)} rows")
    if len({row[10] for row in installed}) != len(installed):
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
    print(f"{len(audit)} source cells, {len(rows)} installed rows, {dict(Counter(a['status'] for a in audit))}")
    if args.install:
        with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
        with AUDIT.open("w", encoding="utf-8") as stream:
            for record in audit:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
