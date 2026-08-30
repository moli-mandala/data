#!/usr/bin/env python3
"""Guard and install the manual Western Indo-Nepal Tharu transcription."""

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[5]
PDF = WORKSPACE / "tmp" / "pdfs" / "tharu_2017" / "silesr2017_013.pdf"
PDF_SHA256 = "a060adebc3c7508b541522ac19b9b7d068ae9ca59c04f7f8e1078eab09e0486c"
INSTALLED = ROOT.parents[1] / "20230530-tharu2.csv"
LEGACY = ROOT / "legacy_20230530_tharu2.csv"
STAGED_AUDIT = ROOT / "staged_audit.tsv"
MANIFEST = ROOT / "source_manifest.json"
EXPECTED_ITEMS = 210
EXPECTED_LISTS = 16
EXPECTED_CELLS = EXPECTED_ITEMS * EXPECTED_LISTS
TARGETS = {
    "BNM", "BNT", "RNK", "RNS_Sisaikhara", "RNS_Sisana", "RKM", "RKB",
    "TkN", "KkP", "SkP", "DKS", "DDK", "DGC", "DkR", "CCC",
}
CONTROL = "HIN"
DECLARATION = "hand-keyed-from-rendered-source; PDF-text-OCR-legacy-not-copied"
METHOD = (
    "manual visual inspection of 400-dpi rendered PDF page; 900/1200-dpi "
    "crops used for every cell; PDF text/OCR/legacy not accepted"
)
STATUSES = {"attested", "source_blank", "ambiguous", "illegible"}
FIELDS = [
    "Item", "Gloss", "Site_Key", "Source_Code", "Source_Code_Occurrence", "Scope",
    "PDF_Page", "Printed_Page", "Column", "Source_Group_Labels",
    "Manual_Transcription", "Manual_Form_Count", "Source_Qualifier", "Review_Status",
    "Confidence", "Site_Assignment_Confidence", "Uncertainty", "Reviewer_Method",
    "Reviewed_At", "Reviewer_Declaration",
]
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Item", "Gloss", "Site_Key", "Scope", "PDF_Page", "Printed_Page", "Column",
    "Source_Group_Labels", "Manual_Transcription", "Review_Status", "Disposition",
    "Reason", "Language_ID", "Dialect_ID", "Source", "Installed_Count",
    "Entry_Keys", "Confidence", "Site_Assignment_Confidence", "Uncertainty",
    "Reviewer_Method", "Reviewer_Declaration",
]
SOURCE_KEY = "webster"
KEY_PREFIX = "webster2017westerntharu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry() -> dict[str, dict[str, str]]:
    with (ROOT / "list_registry.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == EXPECTED_LISTS
    assert [row["Site_Key"] for row in rows][-1] == CONTROL
    assert {row["Site_Key"] for row in rows[:-1]} == TARGETS
    assert Counter(row["Scope"] for row in rows) == Counter(target=15, control=1)
    assert [row["Code_Occurrence"] for row in rows if row["Metadata_Code"] == "RNS"] == ["1", "2"]
    return {row["Site_Key"]: row for row in rows}


def load_cells() -> list[dict[str, str]]:
    registry = load_registry()
    rows = []
    for path in sorted((ROOT / "manual_chunks").glob("items_*_hand_keyed.tsv")):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            assert list(reader.fieldnames or []) == FIELDS, f"unexpected schema: {path}"
            assert not any("ocr" in field.casefold() for field in reader.fieldnames or [])
            rows.extend(reader)
    seen = set()
    for row in rows:
        assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
        item = int(row["Item"])
        assert 1 <= item <= EXPECTED_ITEMS
        assert row["Site_Key"] in registry
        assert row["Scope"] == registry[row["Site_Key"]]["Scope"]
        key = (item, row["Site_Key"])
        assert key not in seen, f"duplicate cell: {key}"
        seen.add(key)
        assert row["Reviewer_Declaration"] == DECLARATION
        assert row["Reviewer_Method"] == METHOD
        assert row["Review_Status"] in STATUSES
        assert row["Confidence"] in {"high", "medium"}
        assert row["Site_Assignment_Confidence"] in {"high", "medium"}
        if row["Site_Key"].startswith("RNS_"):
            assert row["Site_Assignment_Confidence"] == "medium"
            assert "duplicate source code RNS" in row["Uncertainty"]
        if row["Review_Status"] == "attested":
            forms = row["Manual_Transcription"].split(" / ")
            assert all(forms)
            assert int(row["Manual_Form_Count"]) == len(forms)
        else:
            assert not row["Manual_Transcription"]
            assert row["Manual_Form_Count"] == "0"
            assert row["Uncertainty"]
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["Review_Status"] for row in rows)
    return {
        "reviewed_cells": len(rows),
        "attested_cells": counts["attested"],
        "source_blank_cells": counts["source_blank"],
        "ambiguous_cells": counts["ambiguous"],
        "illegible_cells": counts["illegible"],
        "target_reviewed_cells": sum(row["Site_Key"] in TARGETS for row in rows),
        "control_reviewed_cells": sum(row["Site_Key"] == CONTROL for row in rows),
        "target_candidate_forms": sum(
            int(row["Manual_Form_Count"]) for row in rows if row["Site_Key"] in TARGETS
        ),
        "control_candidate_forms": sum(
            int(row["Manual_Form_Count"]) for row in rows if row["Site_Key"] == CONTROL
        ),
        "pending_cells": EXPECTED_CELLS - len(rows),
    }


def expanded_target_counter(rows: list[dict[str, str]]) -> Counter[tuple[str, str, str]]:
    registry = load_registry()
    counter = Counter()
    for row in rows:
        if row["Site_Key"] not in TARGETS or row["Review_Status"] != "attested":
            continue
        legacy_language = registry[row["Site_Key"]]["Proposed_Dialect_ID"]
        if row["Site_Key"].startswith("RNS_"):
            legacy_language = "Tharu-RNS"
        elif row["Site_Key"] == "CCC":
            legacy_language = "Chitwan"
        for form in row["Manual_Transcription"].split(" / "):
            counter[(row["Gloss"], legacy_language, form)] += 1
    return counter


def legacy_reconciliation(rows: list[dict[str, str]]) -> dict[str, int]:
    """Compare only after manual entry; never use legacy strings as source evidence."""
    reviewed_glosses = {row["Gloss"] for row in rows}
    with LEGACY.open(encoding="utf-8", newline="") as handle:
        legacy_rows = [row for row in csv.reader(handle) if row[3] in reviewed_glosses]
    manual = expanded_target_counter(rows)
    legacy = Counter((row[3], row[0], row[2]) for row in legacy_rows)
    exact = manual & legacy
    return {
        "manual_target_occurrences": sum(manual.values()),
        "legacy_target_occurrences": sum(legacy.values()),
        "exact_occurrences": sum(exact.values()),
        "manual_only_occurrences": sum((manual - legacy).values()),
        "legacy_only_occurrences": sum((legacy - manual).values()),
    }


def stage(rows: list[dict[str, str]]) -> None:
    assert len(rows) == EXPECTED_CELLS, (
        f"refusing staging: only {len(rows)}/{EXPECTED_CELLS} cells have manual decisions"
    )
    assert all(row["Review_Status"] in {"attested", "source_blank"} for row in rows), (
        "refusing staging: ambiguous or illegible cells remain"
    )


def dialect_tag(registry_row: dict[str, str]) -> str:
    # Keep the already-public DKS tag spelling stable while retaining the report's
    # exact `Sivratnapur` spelling in the dialect display metadata and Notes.
    tag_location = (
        "Shivratanpur"
        if registry_row["Proposed_Dialect_ID"] == "Tharu-DKS"
        else registry_row["Village"]
    )
    return (
        f"dialect:{registry_row['Parent_Language_ID']}:"
        f"{quote(registry_row['Proposed_Dialect_ID'], safe='')}:"
        f"{quote(tag_location, safe='')}"
    )


def locator(row: dict[str, str]) -> str:
    code = row["Source_Code"]
    if row["Source_Code_Occurrence"]:
        code += f" occurrence {row['Source_Code_Occurrence']}"
    return (
        f"{SOURCE_KEY}[Appendix B, printed p. {row['Printed_Page']} "
        f"(PDF p. {row['PDF_Page']}), item {int(row['Item'])}, {code}]"
    )


def cell_notes(row: dict[str, str], registry_row: dict[str, str]) -> str:
    notes = [
        f"Source list {row['Source_Code']} ({registry_row['Language_Name']}), "
        f"survey site {registry_row['Village']}"
    ]
    if row["Source_Group_Labels"]:
        notes.append(
            "printed lexical-similarity group(s) " + row["Source_Group_Labels"]
            + "; retained as source evidence, not cognacy"
        )
    if row["Source_Qualifier"]:
        notes.append("source qualifier: " + row["Source_Qualifier"])
    if row["Site_Key"].startswith("RNS_"):
        notes.append("dialect assignment uncertain: " + row["Uncertainty"])
    if row["Site_Key"] == "TkN":
        notes.append(
            "source language label Thakur Tharu; existing Rana parent route retained "
            "pending independent classification review"
        )
    if row["Site_Key"] == "RKM":
        notes.append("metadata code RKM; response-table alias RkM")
    return "; ".join(notes)


def build_install(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Expand frozen target cells and account for every target/control cell."""
    stage(rows)
    registry = load_registry()
    installed: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: (int(item["Item"]), list(registry).index(item["Site_Key"]))):
        metadata = registry[row["Site_Key"]]
        source = locator(row)
        keys: list[str] = []
        if row["Scope"] == "target" and row["Review_Status"] == "attested":
            for index, form in enumerate(row["Manual_Transcription"].split(" / "), 1):
                entry_key = (
                    f"{KEY_PREFIX}:item{int(row['Item']):03d}:"
                    f"{row['Site_Key'].lower()}:form{index}"
                )
                tags = [dialect_tag(metadata)]
                if row["Site_Key"].startswith("RNS_"):
                    tags.append("uncertain")
                installed.append({
                    "Language_ID": metadata["Parent_Language_ID"],
                    "Parameter_ID": "", "Form": form, "Gloss": row["Gloss"],
                    "Native": "", "Phonemic": form,
                    "Notes": cell_notes(row, metadata), "Source": source,
                    "Cognateset": "", "Etymology": "", "Entry_Key": entry_key,
                    "Variant_Of_Key": "", "Borrowed_From_Key": "",
                    "Derivation_Parent_Keys": "", "Tags": " ".join(tags),
                })
                keys.append(entry_key)
        if row["Scope"] == "control":
            disposition = "excluded"
            reason = "Standard Hindi comparison control; outside target scope"
        elif row["Review_Status"] == "source_blank":
            disposition = "excluded"
            reason = "source explicitly prints no response for this conceptual cell"
        else:
            disposition = "installed"
            reason = ""
        audit.append({
            "Item": row["Item"], "Gloss": row["Gloss"], "Site_Key": row["Site_Key"],
            "Scope": row["Scope"], "PDF_Page": row["PDF_Page"],
            "Printed_Page": row["Printed_Page"], "Column": row["Column"],
            "Source_Group_Labels": row["Source_Group_Labels"],
            "Manual_Transcription": row["Manual_Transcription"],
            "Review_Status": row["Review_Status"], "Disposition": disposition,
            "Reason": reason, "Language_ID": metadata["Parent_Language_ID"] if row["Scope"] == "target" else "",
            "Dialect_ID": metadata["Proposed_Dialect_ID"], "Source": source,
            "Installed_Count": str(len(keys)), "Entry_Keys": " | ".join(keys),
            "Confidence": row["Confidence"],
            "Site_Assignment_Confidence": row["Site_Assignment_Confidence"],
            "Uncertainty": row["Uncertainty"], "Reviewer_Method": row["Reviewer_Method"],
            "Reviewer_Declaration": row["Reviewer_Declaration"],
        })
    assert len(installed) == 3560
    assert len(audit) == EXPECTED_CELLS
    assert len({row["Entry_Key"] for row in installed}) == len(installed)
    assert Counter(row["Disposition"] for row in audit) == Counter(installed=3052, excluded=308)
    assert sum(int(row["Installed_Count"]) for row in audit) == len(installed)
    assert sum(row["Site_Key"].startswith("RNS_") for row in audit) == 420
    assert all(
        row["Uncertainty"] and row["Site_Assignment_Confidence"] == "medium"
        for row in audit if row["Site_Key"].startswith("RNS_")
    )
    return installed, audit


def install(rows: list[dict[str, str]]) -> dict[str, object]:
    installed, audit = build_install(rows)
    with INSTALLED.open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=FORM_FIELDS, lineterminator="\n").writerows(installed)
    with STAGED_AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(audit)
    return {
        "installed_rows": len(installed),
        "installed_sha256": sha256(INSTALLED),
        "staged_audit_rows": len(audit),
        "staged_audit_sha256": sha256(STAGED_AUDIT),
        "unique_entry_keys": len({row["Entry_Key"] for row in installed}),
        "rns_uncertain_conceptual_cells": sum(
            row["Site_Key"].startswith("RNS_") for row in audit
        ),
        "rns_uncertain_installed_forms": sum(
            "uncertain" in row["Tags"].split() for row in installed
        ),
        "target_blank_cells_excluded": sum(
            row["Scope"] == "target" and row["Review_Status"] == "source_blank"
            for row in audit
        ),
        "hindi_control_cells_excluded": sum(row["Scope"] == "control" for row in audit),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    assert sha256(PDF) == PDF_SHA256, "canonical PDF hash mismatch"
    rows = load_cells()
    summary = summarize(rows)
    summary["legacy_reconciliation"] = legacy_reconciliation(rows)
    if args.stage:
        stage(rows)
    if args.install:
        summary["integration"] = install(rows)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
