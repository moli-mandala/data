#!/usr/bin/env python3
"""Validate and stage the manually reviewed Kurumba Appendix C wordlists.

This importer deliberately ignores every ``OCR_*`` field.  It will not emit
forms while a cell, page, or prompt remains pending, and it excludes unresolved
ambiguous/illegible cells rather than guessing a reading.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[5]
PDF = WORKSPACE_ROOT / "tmp/pdfs/kurumba_2012/silesr2012_015.pdf"
LISTS = HERE / "list_registry.tsv"
PAGES = HERE / "page_review.tsv"
PROMPTS = HERE / "prompt_review.tsv"
MANUAL = HERE / "manual_transcription.tsv"
MANUAL_CHUNKS = HERE / "manual_chunks"
UNRESOLVED = HERE / "unresolved_readings.tsv"
STAGED_FORMS = HERE / "staged_forms.csv"
STAGED_AUDIT = HERE / "staged_audit.csv"
MANIFEST = HERE / "source_manifest.json"

SOURCE_KEY = "blairetal2012kurumba"
PDF_SHA256 = "250dc3d83661227caa66bf16e390e51c2dcb7186fa435252541ed13bbfcd9137"
REVIEW_METHOD = "manual visual transcription from rendered source scan; OCR used only as locator/comparison scaffold"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Cell_Key", "PDF_Page", "Printed_Page", "Row", "Item", "List_Key",
    "Scope", "Language_ID", "Dialect_ID", "Manual_Gloss", "Manual_Form",
    "Cell_Status", "Confidence", "Review_Method", "Status", "Reason",
    "Citation", "Entry_Key",
]
CHUNK_FIELDS = {
    "Cell_Key", "Manual_Form", "Cell_Status", "Confidence",
    "Review_Method", "Reviewer", "Notes",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_manual_chunks(cells: list[dict[str, str]]) -> list[dict[str, str]]:
    """Overlay disjoint, hand-reviewed page chunks without mutating the base ledger.

    Chunks make parallel visual transcription safe: every accepted cell keeps its
    stable key, a cell may occur in at most one chunk, and chunks may replace only
    still-pending base rows.  OCR scaffold fields are neither required nor read.
    """
    by_key = {row["Cell_Key"]: row for row in cells}
    seen: set[str] = set()
    for path in sorted(MANUAL_CHUNKS.glob("*.tsv")):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if set(reader.fieldnames or ()) != CHUNK_FIELDS:
                raise ValueError(f"Unexpected review-chunk columns in {path.name}")
            for patch in reader:
                key = patch["Cell_Key"]
                if key in seen:
                    raise ValueError(f"Cell occurs in more than one review chunk: {key}")
                seen.add(key)
                if key not in by_key:
                    raise ValueError(f"Unknown review-chunk cell key: {key}")
                row = by_key[key]
                if row["Cell_Status"] != "pending":
                    raise ValueError(f"Review chunk would overwrite completed base cell: {key}")
                for field in CHUNK_FIELDS - {"Cell_Key"}:
                    row[field] = patch[field]
    return cells


def validate_topology() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    lists = rows(LISTS)
    pages = rows(PAGES)
    prompts = rows(PROMPTS)
    cells = apply_manual_chunks(rows(MANUAL))
    if len(lists) != 19 or Counter(row["Scope"] for row in lists) != {"target": 15, "control": 4}:
        raise ValueError("Expected 19 lists: 15 targets and 4 comparison controls")
    if len(pages) != 220 or {int(row["PDF_Page"]) for row in pages} != set(range(217, 437)):
        raise ValueError("Expected all 220 data pages, physical PDF 217-436")
    if sum(int(row["Conceptual_Cells"]) for row in pages) != 10450:
        raise ValueError("Page ledger does not total 10,450 conceptual cells")
    if len(prompts) != 550 or {int(row["Item"]) for row in prompts} != set(range(1, 551)):
        raise ValueError("Expected one review row for each of 550 prompts")
    if len(cells) != 10450 or len({row["Cell_Key"] for row in cells}) != 10450:
        raise ValueError("Expected 10,450 unique manual cell rows")
    expected = {(row["List_Key"], str(item)) for row in lists for item in range(1, 551)}
    actual = {(row["List_Key"], row["Item"]) for row in cells}
    if actual != expected:
        raise ValueError("Manual ledger does not cover every list/item tuple")
    return lists, pages, prompts, cells


def completion_counts(pages: list[dict[str, str]], prompts: list[dict[str, str]], cells: list[dict[str, str]]) -> dict[str, int]:
    cell_counts = Counter(row["Cell_Status"] for row in cells)
    return {
        "pages_pending": sum(row["Review_Status"] != "complete" for row in pages),
        "prompts_pending": sum(row["Review_Status"] != "complete" for row in prompts),
        "cells_pending": cell_counts["pending"],
        "cells_attested": cell_counts["attested"],
        "cells_blank": cell_counts["blank"],
        "cells_ambiguous": cell_counts["ambiguous"],
        "cells_illegible": cell_counts["illegible"],
    }


def require_manual_completion(pages: list[dict[str, str]], prompts: list[dict[str, str]], cells: list[dict[str, str]]) -> None:
    allowed = {"attested", "blank", "ambiguous", "illegible"}
    pending = [row for row in cells if row["Cell_Status"] == "pending"]
    unknown = [row for row in cells if row["Cell_Status"] not in allowed | {"pending"}]
    if unknown:
        raise ValueError(f"Unknown cell status in {unknown[0]['Cell_Key']}")
    if pending or any(row["Review_Status"] != "complete" for row in pages) or any(row["Review_Status"] != "complete" for row in prompts):
        raise RuntimeError("manual visual review incomplete")
    if any(row["Review_Method"] != REVIEW_METHOD for row in cells):
        raise ValueError("Every cell must record the approved manual visual review method")
    for row in prompts:
        if not row["Manual_Gloss"] or not row["Confidence"]:
            raise ValueError(f"Prompt {row['Item']} lacks a manually reviewed gloss/confidence")
    for row in cells:
        status, form = row["Cell_Status"], row["Manual_Form"]
        if status == "attested" and (not form or not row["Confidence"]):
            raise ValueError(f"Attested cell lacks manual form/confidence: {row['Cell_Key']}")
        if status in {"blank", "illegible"} and form:
            raise ValueError(f"{status} cell must not invent a form: {row['Cell_Key']}")
        if status in {"ambiguous", "illegible"} and not row["Notes"]:
            raise ValueError(f"Unresolved cell lacks a note: {row['Cell_Key']}")


def build(lists: list[dict[str, str]], prompts: list[dict[str, str]], cells: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    by_list = {row["List_Key"]: row for row in lists}
    glosses = {row["Item"]: row["Manual_Gloss"] for row in prompts}
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for row in cells:
        spec = by_list[row["List_Key"]]
        status = row["Cell_Status"]
        gloss = glosses[row["Item"]]
        citation = (
            f"{SOURCE_KEY}[Appendix C, printed p. {row['Printed_Page']}, "
            f"item {row['Item']}, list {spec['Source_Label']}]"
        )
        entry_key = ""
        audit_status = "missing" if status == "blank" else "unresolved"
        reason = row["Notes"]
        if status == "attested":
            form = unicodedata.normalize("NFC", row["Manual_Form"])
            entry_key = row["Cell_Key"].replace("kurumba2012:", "kurumba2012-").replace(":", "-")
            display = spec["Source_Label"]
            dialect_tag = f"dialect:{spec['Language_ID']}:{spec['Dialect_ID']}:{quote(display)}"
            notes = f"manually transcribed from scan; source list classification: {spec['Report_Classification']}"
            forms.append([
                spec["Language_ID"], "", form, gloss, "", form, notes, citation,
                "", "", entry_key, "", "", "", dialect_tag,
            ])
            audit_status = "staged"
        audit.append(dict(zip(AUDIT_FIELDS, [
            row["Cell_Key"], row["PDF_Page"], row["Printed_Page"], row["Row"],
            row["Item"], row["List_Key"], row["Scope"], row["Language_ID"],
            row["Dialect_ID"], gloss, row["Manual_Form"], status,
            row["Confidence"], row["Review_Method"], audit_status, reason,
            citation, entry_key,
        ])))
    return forms, audit


def write(forms: list[list[str]], audit: list[dict[str, str]], counts: dict[str, int]) -> None:
    with STAGED_FORMS.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(forms)
    with STAGED_AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    unresolved = [row for row in audit if row["Status"] == "unresolved"]
    manifest = {
        "source_key": SOURCE_KEY,
        "pdf_sha256": PDF_SHA256,
        "pdf_bytes": PDF.stat().st_size,
        "pdf_pages": 436,
        "appendix_data_pdf_pages": [217, 436],
        "lists": 19,
        "target_lists": 15,
        "comparison_lists": 4,
        "prompts": 550,
        "conceptual_cells": 10450,
        **counts,
        "installed_forms": len(forms),
        "unresolved_readings": len(unresolved),
        "review_authority": REVIEW_METHOD,
        "ocr_authority": "none; OCR fields are never read by this importer",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-pdf", action="store_true")
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    if args.verify_pdf:
        if not PDF.exists() or sha256(PDF) != PDF_SHA256:
            raise SystemExit("Canonical PDF missing or checksum mismatch")
    lists, pages, prompts, cells = validate_topology()
    counts = completion_counts(pages, prompts, cells)
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    try:
        require_manual_completion(pages, prompts, cells)
    except RuntimeError as error:
        if args.stage:
            raise SystemExit(f"Refusing to stage: {error}")
        return
    forms, audit = build(lists, prompts, cells)
    if args.stage:
        write(forms, audit, counts)
    print(f"review_complete=1 forms={len(forms)} audit_rows={len(audit)}")


if __name__ == "__main__":
    main()
