#!/usr/bin/env python3
"""Validate, overlay, and stage the manually transcribed Ho Appendix D.3.

OCR is deliberately outside the accepted-data path. Review chunks must be
OCR-blind and carry an explicit hand-keying declaration. The importer refuses
to stage until all 5,670 source cells have a final manual-review status.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[5]
PDF = WORKSPACE_ROOT / "tmp/pdfs/ho_2024/JLSR2024_009.pdf"
BASE = HERE / "manual_review.tsv"
LISTS = HERE / "list_registry.tsv"
CHUNKS = HERE / "manual_chunks"
UNRESOLVED = HERE / "unresolved_readings.tsv"
STAGED_FORMS = HERE / "staged_forms.csv"
STAGED_AUDIT = HERE / "staged_audit.tsv"
MANIFEST = HERE / "source_manifest.json"

SOURCE_KEY = "varenkamp2024ho"
PDF_SHA256 = "5ca30882dc5ed0f8480c9710e5fc0e08bf4d92e27d591582e3d953709ec1f9d1"
SITES = "HO1 HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HO2 HRA HO3 HOP HBA HNI BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
TARGETS = "HTH HKA HKE HCH HCU HSU HSA HJO HDH HBG HRA HOP HBA HNI".split()
REPUBLISHED = "HO1 HO2 HO3".split()
COMPARATORS = "BBG BMA BOP BRA BGH MU1 MU2 SA1 SBA OCU".split()
FINAL_STATUSES = {"attested", "blank", "ambiguous", "illegible"}
BASE_FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "OCR_Evidence_Only", "Manual_Transcription", "Review_Status", "Confidence",
    "Uncertainty", "Reviewer_Method", "Reviewed_At",
    "Reviewer_Declaration",
]
CHUNK_FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Manual_Transcription", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
AUDIT_FIELDS = [
    "Item", "Gloss", "Site_Code", "Scope", "PDF_Page", "Printed_Page", "Column",
    "Manual_Transcription", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Disposition", "Citation", "Entry_Key",
]
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader.fieldnames or ()), list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_base() -> list[dict[str, str]]:
    fields, rows = read_tsv(BASE)
    if fields != BASE_FIELDS:
        raise ValueError("Unexpected manual_review.tsv columns")
    if len(rows) != 5670:
        raise ValueError(f"Expected 5,670 base cells, found {len(rows)}")
    expected = {(str(item), site) for item in range(1, 211) for site in SITES}
    actual = [(row["Item"], row["Site_Code"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Base ledger must contain every unique Item+Site_Code key")
    for row in rows:
        item = int(row["Item"]); page = int(row["PDF_Page"])
        if page != 72 + (item - 1) // 3 or int(row["Printed_Page"]) != page - 9:
            raise ValueError(f"Coordinate mismatch for {item}+{row['Site_Code']}")
        expected_column = "left" if SITES.index(row["Site_Code"]) < 14 else "right"
        if row["Column"] != expected_column:
            raise ValueError(f"Column mismatch for {item}+{row['Site_Code']}")
        if not all(unicodedata.is_normalized("NFC", value) for value in row.values()):
            raise ValueError(f"Non-NFC base row: {item}+{row['Site_Code']}")
    return rows


def chunk_paths() -> list[Path]:
    return sorted(
        path for path in CHUNKS.glob("pages_*.tsv")
        if not path.name.endswith("_unresolved.tsv")
    )


def overlay_manual_chunks(base_rows: list[dict[str, str]], paths: list[Path] | None = None) -> list[dict[str, str]]:
    """Overlay disjoint OCR-blind chunks only onto unreviewed base cells."""
    rows = deepcopy(base_rows)
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    patched: set[tuple[str, str]] = set()
    for path in chunk_paths() if paths is None else paths:
        fields, chunks = read_tsv(path)
        if any(field.startswith("OCR") for field in fields):
            raise ValueError(f"OCR-bearing review chunk is inadmissible: {path.name}")
        if fields != CHUNK_FIELDS:
            raise ValueError(f"Unexpected OCR-blind review-chunk columns: {path.name}")
        for patch in chunks:
            key = (patch["Item"], patch["Site_Code"])
            if key in patched:
                raise ValueError(f"Duplicate review-chunk key: {'+'.join(key)}")
            if key not in by_key:
                raise ValueError(f"Unknown review-chunk key: {'+'.join(key)}")
            row = by_key[key]
            if row["Review_Status"] != "unreviewed":
                raise ValueError(f"Chunk overlaps reviewed base row: {'+'.join(key)}")
            for field in ("PDF_Page", "Printed_Page", "Column"):
                if patch[field] != row[field]:
                    raise ValueError(f"Chunk coordinate mismatch for {'+'.join(key)}: {field}")
            if not patch["Gloss"] or (row["Gloss"] and patch["Gloss"] != row["Gloss"]):
                raise ValueError(f"Chunk gloss mismatch for {'+'.join(key)}")
            if patch["Reviewer_Declaration"] != DECLARATION:
                raise ValueError(f"Missing hand-keying declaration for {'+'.join(key)}")
            if not patch["Reviewer_Method"].startswith("manual-source-image; rendered-") or not patch["Reviewer_Method"].endswith("; OCR-not-accepted"):
                raise ValueError(f"Unapproved review method for {'+'.join(key)}")
            status, form = patch["Review_Status"], patch["Manual_Transcription"]
            if status not in FINAL_STATUSES:
                raise ValueError(f"Non-final chunk status for {'+'.join(key)}: {status}")
            if status in {"attested", "ambiguous"} and not form:
                raise ValueError(f"{status} chunk cell lacks a manual form: {'+'.join(key)}")
            if status in {"blank", "illegible"} and form:
                raise ValueError(f"{status} chunk cell invents a form: {'+'.join(key)}")
            if status in {"ambiguous", "illegible"} and not patch["Uncertainty"]:
                raise ValueError(f"Unresolved chunk cell lacks a note: {'+'.join(key)}")
            if not patch["Reviewed_At"]:
                raise ValueError(f"Chunk cell lacks review date: {'+'.join(key)}")
            if not all(unicodedata.is_normalized("NFC", value) for value in patch.values()):
                raise ValueError(f"Non-NFC chunk row: {'+'.join(key)}")
            for field in CHUNK_FIELDS:
                row[field] = patch[field]
            patched.add(key)
    return rows


def validate_registry() -> list[dict[str, str]]:
    _, specs = read_tsv(LISTS)
    if len(specs) != 27 or [row["Site_Code"] for row in specs] != SITES:
        raise ValueError("List registry must preserve all 27 source rows in order")
    if Counter(row["Scope"] for row in specs) != Counter(target=14, republished_control=3, comparison_control=10):
        raise ValueError("Expected 14 target, 3 republished, and 10 comparator lists")
    if {row["Site_Code"] for row in specs if row["Install"] == "yes"} != set(TARGETS):
        raise ValueError("Only the fourteen new 1989 Ho field lists may install")
    return specs


def validate_effective(rows: list[dict[str, str]]) -> Counter:
    counts = Counter(row["Review_Status"] for row in rows)
    unknown = set(counts) - FINAL_STATUSES - {"unreviewed"}
    if unknown:
        raise ValueError(f"Unknown review statuses: {sorted(unknown)}")
    for row in rows:
        status, form = row["Review_Status"], row["Manual_Transcription"]
        key = f"{row['Item']}+{row['Site_Code']}"
        if status in {"attested", "ambiguous"} and not form:
            raise ValueError(f"{status} cell lacks manual transcription: {key}")
        if status in {"blank", "illegible", "unreviewed"} and form:
            raise ValueError(f"{status} cell must have no accepted form: {key}")
        if status in FINAL_STATUSES:
            method = row["Reviewer_Method"]
            if not method.startswith("manual-source-image; rendered-") or not method.endswith("; OCR-not-accepted"):
                raise ValueError(f"Final cell lacks manual-method stamp: {key}")
            if row["Reviewer_Declaration"] != DECLARATION:
                raise ValueError(f"Final cell lacks exact hand-keying declaration: {key}")
        if status in {"ambiguous", "illegible"} and not row["Uncertainty"]:
            raise ValueError(f"Unresolved cell lacks explanation: {key}")
    return counts


def require_complete(rows: list[dict[str, str]]) -> Counter:
    counts = validate_effective(rows)
    if counts["unreviewed"]:
        raise RuntimeError(f"manual visual review incomplete: {counts['unreviewed']} of 5,670 cells unreviewed")
    if sum(counts[status] for status in FINAL_STATUSES) != 5670:
        raise RuntimeError("manual visual review does not account for all 5,670 cells")
    return counts


def strip_similarity_labels(text: str) -> str:
    """Remove source similarity-group labels while preserving cell punctuation."""
    value = re.sub(r"(^|,\s*)\d+(?=\s|,|\()\s*", r"\1", text)
    value = re.sub(r"^,\s*", "", value)
    return value.strip()


def build(rows: list[dict[str, str]], specs: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    """Build target-only staging rows solely from Manual_Transcription."""
    by_site = {row["Site_Code"]: row for row in specs}
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for row in rows:
        spec = by_site[row["Site_Code"]]
        status = row["Review_Status"]
        citation = f"{SOURCE_KEY}[Appendix D.3, printed p. {row['Printed_Page']}, item {row['Item']}, list {row['Site_Code']}]"
        disposition = "control-excluded" if spec["Install"] != "yes" else "missing"
        entry_key = ""
        if spec["Install"] == "yes" and status == "attested":
            form = strip_similarity_labels(row["Manual_Transcription"])
            if not form:
                raise ValueError(f"Similarity-label removal emptied target cell {row['Item']}+{row['Site_Code']}")
            entry_key = f"ho2024-{row['Site_Code'].lower()}-i{int(row['Item']):03d}"
            tag = f"dialect:ho:{row['Site_Code']}:{quote(spec['Label'])}"
            notes = "1989 field list; diplomatic manual transcription; comma-separated source alternatives preserved"
            forms.append(["ho", "", form, row["Gloss"], "", form, notes, citation, "", "", entry_key, "", "", "", tag])
            disposition = "staged"
        elif spec["Install"] == "yes" and status in {"ambiguous", "illegible"}:
            disposition = "unresolved-excluded"
        audit.append(dict(zip(AUDIT_FIELDS, [
            row["Item"], row["Gloss"], row["Site_Code"], spec["Scope"], row["PDF_Page"],
            row["Printed_Page"], row["Column"], row["Manual_Transcription"], status,
            row["Confidence"], row["Uncertainty"], row["Reviewer_Method"], disposition,
            citation, entry_key,
        ])))
    return forms, audit


def write_unresolved(audit: list[dict[str, str]]) -> None:
    rows = [row for row in audit if row["Review_Status"] in {"ambiguous", "illegible"}]
    with UNRESOLVED.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def stage(forms: list[list[str]], audit: list[dict[str, str]]) -> None:
    with STAGED_FORMS.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(forms)
    with STAGED_AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(audit)
    write_unresolved(audit)


def role_counts(rows: list[dict[str, str]]) -> dict[str, Counter]:
    roles = {"target": set(TARGETS), "republished_control": set(REPUBLISHED), "comparison_control": set(COMPARATORS)}
    return {role: Counter(row["Review_Status"] for row in rows if row["Site_Code"] in sites) for role, sites in roles.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-pdf", action="store_true")
    parser.add_argument("--base-only", action="store_true", help="diagnose base ledger without applying chunks; cannot stage")
    parser.add_argument("--write-unresolved", action="store_true", help="refresh unresolved_readings.tsv from admissible reviewed rows")
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    if args.verify_pdf and (not PDF.exists() or PDF.stat().st_size != 12_467_726 or sha256(PDF) != PDF_SHA256):
        raise SystemExit("Canonical PDF missing or checksum mismatch")
    if args.base_only and args.stage:
        raise SystemExit("Refusing to stage: --base-only bypasses review chunks")
    base = validate_base(); specs = validate_registry()
    rows = base if args.base_only else overlay_manual_chunks(base)
    counts = validate_effective(rows)
    print(" ".join(f"cells_{status}={counts[status]}" for status in ["attested", "blank", "ambiguous", "illegible", "unreviewed"]))
    if args.write_unresolved:
        _, audit = build(rows, specs)
        write_unresolved(audit)
    if args.stage:
        try: require_complete(rows)
        except RuntimeError as error: raise SystemExit(f"Refusing to stage: {error}")
        forms, audit = build(rows, specs); stage(forms, audit)
        print(f"review_complete=1 staged_forms={len(forms)} audit_rows={len(audit)}")


if __name__ == "__main__": main()
