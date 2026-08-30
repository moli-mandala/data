#!/usr/bin/env python3
"""Guard and stage Northern Dhule Bhils Appendix C manual review.

OCR is deliberately absent from the accepted-data path.  Review chunks must
be OCR-blind and carry the exact declaration that the response was hand-keyed
from the rendered source.  Staging is refused until all 2,730 cells have a
final manual status and the target-list duplicate/identity plan is resolved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[5]
PDF = WORKSPACE_ROOT / "tmp/pdfs/northern_dhule_bhils_2013/silesr2013_004.pdf"
BASE = HERE / "manual_review.tsv"
LISTS = HERE / "list_registry.tsv"
CHUNKS = HERE / "manual_chunks"
UNRESOLVED = HERE / "unresolved_readings.tsv"
STAGED_FORMS = HERE / "staged_forms.csv"
STAGED_AUDIT = HERE / "staged_audit.tsv"

SOURCE_KEY = "watters2013northerndhule"
PDF_SHA256 = "edeeeda98cb76624df1a0d70c765cc816ea463d75bc79ec20883c62e6fc1c482"
SITES = "KEL DHA DIG AMO MUN AST MAN BHU AML SEG KAN SHA TOR".split()
TARGETS = set(SITES[:-1])
FINAL_STATUSES = {"attested", "blank", "ambiguous", "illegible"}
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
FIELDS = [
    "Item", "Gloss", "Site_Code", "PDF_Page", "Printed_Page", "Column",
    "Manual_Transcription", "Review_Status", "Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewed_At", "Reviewer_Declaration",
]
AUDIT_FIELDS = FIELDS + ["Scope", "Disposition", "Citation"]
RAW_FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
DIALECT_TAGS = {
    "KEL": "dialect:Vasavi:sil-dhule-2013-vasavi-kelpada:Kelpada",
    "DHA": "dialect:Vasavi:sil-dhule-2013-vasavi-dhanoura:Dhanoura",
    "DIG": "dialect:Vasavi:sil-dhule-2013-vasavi-digiamba:Digiamba",
    "AMO": "dialect:Vasavi:sil-dhule-2013-vasavi-amoda:Amoda",
    "MUN": "dialect:Noiri:sil-dhule-2013-noiri-mundalwad:Mundalwad",
    "AST": "dialect:Noiri:sil-dhule-2013-noiri-astamba:Astamba",
    "MAN": "dialect:PauriBareli:sil-bareli-2018-bareli-pauri-mandvi:Mandvi",
    "BHU": "dialect:PauriBareli:sil-dhule-2013-pauri-bhusha:Bhusha",
    "AML": "dialect:RathwiBareli:sil-bareli-2018-rathwi-pauri-amalwadi:Amalwadi",
    "SEG": "dialect:RathwiBareli:sil-bareli-2018-rathwi-pauri-segwi:Segwi",
    "KAN": "dialect:RathwiBareli:sil-dhule-2013-rathwi-kangai:Kangai",
    "SHA": "dialect:PauriBareli:sil-bareli-2018-bareli-pauri-shahana:Shahana",
}


def page_for(item: int) -> int:
    if item <= 45:
        return 91 + (item - 1) // 5
    if item <= 49:
        return 100
    if item <= 194:
        return 101 + (item - 50) // 5
    if item <= 199:
        return 130
    if item <= 204:
        return 131
    if item <= 209:
        return 132
    return 133


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
    if fields != FIELDS:
        raise ValueError("Unexpected manual_review.tsv columns")
    if len(rows) != 2730:
        raise ValueError(f"Expected 2,730 base cells, found {len(rows)}")
    expected = {(str(item), site) for item in range(1, 211) for site in SITES}
    actual = [(row["Item"], row["Site_Code"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Base ledger must contain every unique Item+Site_Code key")
    for row in rows:
        item = int(row["Item"]); page = int(row["PDF_Page"])
        key = f"{item}+{row['Site_Code']}"
        if page != page_for(item) or int(row["Printed_Page"]) != page - 8:
            raise ValueError(f"Coordinate mismatch for {key}")
        column = "left" if SITES.index(row["Site_Code"]) < 6 else "right"
        if row["Column"] != column:
            raise ValueError(f"Column mismatch for {key}")
        if not all(unicodedata.is_normalized("NFC", value) for value in row.values()):
            raise ValueError(f"Non-NFC base row: {key}")
    return rows


def chunk_paths() -> list[Path]:
    return sorted(CHUNKS.glob("items_*_hand_keyed.tsv"))


def overlay_manual_chunks(base_rows: list[dict[str, str]], paths: list[Path] | None = None) -> list[dict[str, str]]:
    rows = deepcopy(base_rows)
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    patched: set[tuple[str, str]] = set()
    for path in chunk_paths() if paths is None else paths:
        fields, chunks = read_tsv(path)
        if any(field.upper().startswith("OCR") for field in fields):
            raise ValueError(f"OCR-bearing review chunk is inadmissible: {path.name}")
        if fields != FIELDS:
            raise ValueError(f"Unexpected OCR-blind review-chunk columns: {path.name}")
        for patch in chunks:
            key = (patch["Item"], patch["Site_Code"])
            printable = "+".join(key)
            if key in patched:
                raise ValueError(f"Duplicate review-chunk key: {printable}")
            if key not in by_key:
                raise ValueError(f"Unknown review-chunk key: {printable}")
            row = by_key[key]
            if row["Review_Status"] != "unreviewed":
                raise ValueError(f"Chunk overlaps reviewed base row: {printable}")
            for field in ("PDF_Page", "Printed_Page", "Column"):
                if patch[field] != row[field]:
                    raise ValueError(f"Chunk coordinate mismatch for {printable}: {field}")
            if not patch["Gloss"]:
                raise ValueError(f"Chunk lacks source gloss for {printable}")
            if patch["Reviewer_Declaration"] != DECLARATION:
                raise ValueError(f"Missing exact hand-keying declaration for {printable}")
            method = patch["Reviewer_Method"]
            if not method.startswith("manual-source-image; rendered-") or not method.endswith("; OCR-not-accepted"):
                raise ValueError(f"Unapproved review method for {printable}")
            status, form = patch["Review_Status"], patch["Manual_Transcription"]
            if status not in FINAL_STATUSES:
                raise ValueError(f"Non-final review status for {printable}: {status}")
            if status in {"attested", "ambiguous"} and not form:
                raise ValueError(f"{status} cell lacks manual transcription: {printable}")
            if status in {"blank", "illegible"} and form:
                raise ValueError(f"{status} cell invents a form: {printable}")
            if status in {"ambiguous", "illegible"} and not patch["Uncertainty"]:
                raise ValueError(f"Unresolved cell lacks explanation: {printable}")
            if not patch["Reviewed_At"]:
                raise ValueError(f"Reviewed cell lacks review date: {printable}")
            if not all(unicodedata.is_normalized("NFC", value) for value in patch.values()):
                raise ValueError(f"Non-NFC chunk row: {printable}")
            row.update(patch)
            patched.add(key)
    return rows


def validate_registry() -> list[dict[str, str]]:
    fields, rows = read_tsv(LISTS)
    expected_fields = ["Site_Code", "Scope", "Install", "Language_ID", "Source_Type", "Label"]
    if fields != expected_fields or [row["Site_Code"] for row in rows] != SITES:
        raise ValueError("List registry must preserve all thirteen source lists in order")
    if Counter(row["Scope"] for row in rows) != Counter(target=12, comparison_control=1):
        raise ValueError("Expected twelve target lists and one comparison control")
    if any(row["Install"] != "no" for row in rows if row["Scope"] == "comparison_control"):
        raise ValueError("Toranmal control may not install from this lane")
    for row in rows:
        if row["Scope"] == "target":
            if row["Install"] != "yes" or not row["Language_ID"]:
                raise ValueError(f"Target route is incomplete: {row['Site_Code']}")
            if row["Site_Code"] not in DIALECT_TAGS:
                raise ValueError(f"Target lacks dialect route: {row['Site_Code']}")
    return rows


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
            if row["Reviewer_Declaration"] != DECLARATION:
                raise ValueError(f"Final cell lacks exact hand-keying declaration: {key}")
            method = row["Reviewer_Method"]
            if not method.startswith("manual-source-image; rendered-") or not method.endswith("; OCR-not-accepted"):
                raise ValueError(f"Final cell lacks manual-method stamp: {key}")
        if status in {"ambiguous", "illegible"} and not row["Uncertainty"]:
            raise ValueError(f"Unresolved cell lacks explanation: {key}")
    return counts


def require_complete(rows: list[dict[str, str]]) -> Counter:
    counts = validate_effective(rows)
    if counts["unreviewed"]:
        raise RuntimeError(f"manual visual review incomplete: {counts['unreviewed']} of 2,730 cells unreviewed")
    if sum(counts[status] for status in FINAL_STATUSES) != 2730:
        raise RuntimeError("manual review does not account for all 2,730 cells")
    return counts


def strip_similarity_labels(text: str) -> str:
    value = re.sub(r"(^|,\s*)\d+(?=\s|,|\()\s*", r"\1", text)
    return re.sub(r"^,\s*", "", value).strip()


def build_audit(rows: list[dict[str, str]], specs: list[dict[str, str]]) -> list[dict[str, str]]:
    by_site = {row["Site_Code"]: row for row in specs}
    audit = []
    for row in rows:
        spec = by_site[row["Site_Code"]]
        if spec["Scope"] != "target":
            disposition = "control-excluded"
        elif row["Review_Status"] in {"ambiguous", "illegible"}:
            disposition = "unresolved-excluded"
        elif row["Review_Status"] == "blank":
            disposition = "blank-excluded"
        elif row["Review_Status"] == "attested":
            disposition = "staged"
        else:
            disposition = "unreviewed-excluded"
        citation = (
            f"{SOURCE_KEY}[Appendix C, printed p. {row['Printed_Page']}, "
            f"item {row['Item']}, list {row['Site_Code']}]"
        )
        audit.append({**row, "Scope": spec["Scope"], "Disposition": disposition, "Citation": citation})
    return audit


def write_unresolved(audit: list[dict[str, str]]) -> None:
    unresolved = [row for row in audit if row["Review_Status"] in {"ambiguous", "illegible"}]
    with UNRESOLVED.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(unresolved)


def staged_rows(rows: list[dict[str, str]], specs: list[dict[str, str]]) -> list[dict[str, str]]:
    require_complete(rows)
    by_site = {row["Site_Code"]: row for row in specs}
    output = []
    for row in rows:
        spec = by_site[row["Site_Code"]]
        if spec["Scope"] != "target" or spec["Install"] != "yes":
            continue
        if row["Review_Status"] != "attested":
            continue
        form = strip_similarity_labels(row["Manual_Transcription"])
        if not form:
            raise ValueError(f"Attested cell becomes empty after label stripping: {row['Item']}+{row['Site_Code']}")
        citation = (
            f"{SOURCE_KEY}[Appendix C, printed p. {row['Printed_Page']}, "
            f"item {row['Item']}, list {row['Site_Code']}]"
        )
        output.append({
            "Language_ID": spec["Language_ID"], "Parameter_ID": "",
            "Form": form, "Gloss": row["Gloss"], "Native": "", "Phonemic": "",
            "Notes": "", "Source": citation, "Cognateset": "", "Etymology": "",
            "Entry_Key": f"{SOURCE_KEY}:item:{int(row['Item']):03d}:site:{row['Site_Code']}",
            "Variant_Of_Key": "", "Borrowed_From_Key": "",
            "Derivation_Parent_Keys": "", "Tags": DIALECT_TAGS[row["Site_Code"]],
        })
    if len(output) != 2497:
        raise ValueError(f"Expected 2,497 staged target attestations, found {len(output)}")
    if len({row["Entry_Key"] for row in output}) != len(output):
        raise ValueError("Duplicate staged Entry_Key")
    if not all(unicodedata.is_normalized("NFC", value) for row in output for value in row.values()):
        raise ValueError("Non-NFC staged output")
    return output


def stage(rows: list[dict[str, str]], specs: list[dict[str, str]]) -> None:
    output = staged_rows(rows, specs)
    audit = build_audit(rows, specs)
    with STAGED_FORMS.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FORM_FIELDS, lineterminator="\n")
        writer.writerows(output)
    with STAGED_AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(audit)
    print(f"staged_forms={len(output)} staged_audit={len(audit)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-pdf", action="store_true")
    parser.add_argument("--write-unresolved", action="store_true")
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    if args.verify_pdf and (
        not PDF.exists() or PDF.stat().st_size != 9_214_722 or sha256(PDF) != PDF_SHA256
    ):
        raise SystemExit("Canonical PDF missing or checksum mismatch")
    base = validate_base(); specs = validate_registry()
    effective = overlay_manual_chunks(base)
    counts = validate_effective(effective)
    print(" ".join(f"cells_{status}={counts[status]}" for status in ["attested", "blank", "ambiguous", "illegible", "unreviewed"]))
    audit = build_audit(effective, specs)
    if args.write_unresolved:
        write_unresolved(audit)
    if args.stage:
        try:
            stage(effective, specs)
        except RuntimeError as error:
            raise SystemExit(f"Refusing to stage: {error}")


if __name__ == "__main__":
    main()
