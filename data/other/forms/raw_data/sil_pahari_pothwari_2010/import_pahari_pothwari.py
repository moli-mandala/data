#!/usr/bin/env python3
"""Install the fourteen target wordlists in SIL ESR 2010-012."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
FORMS = REPO / "data/other/forms"
RAW_ROOT = FORMS / "raw_data"
SNAPSHOT = HERE / "wordlist_snapshot.tsv"
INSTALLED = FORMS / "20260828-sil-pahari-pothwari.csv"
AUDIT = RAW_ROOT / "20260828-sil-pahari-pothwari-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-pahari-pothwari-manifest.json"

SOURCE_KEY = "lothers-lothers2010pahari"
SOURCE_SHA256 = "e3695a807c4856118303eca74b68b192817ea69251fa8be62abb7b27e4c1ad6f"
SNAPSHOT_SHA256 = "9ef7a0f32c9b2d1d263c1d0fba213d9db67bf1927237915320227c1c6492e7e1"
LANGUAGE_ID = "poth"
ENTRY_PREFIX = "silpaharipothwari2010"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Source_Key", "PDF_Page", "Printed_Page", "Item", "Gloss",
    "Excluded_From_Similarity", "Raw_Lect_Code", "Lect_Code", "Site", "District",
    "Reliability", "Raw_Form", "Form", "Review", "Source_Scope", "Source_Status",
    "Status", "Reason", "Language_ID", "Dialect_ID", "Citation", "Entry_Key",
]


def dialect_id(code: str) -> str:
    return f"sil-pahari-pothwari-2010-{code.lower()}"


def dialect_tag(code: str, site: str) -> str:
    did = dialect_id(code)
    return f"dialect:{LANGUAGE_ID}:{quote(did, safe='')}:{quote(site + '-Pahari-Pothwari', safe='')}"


def citation(row: dict[str, str]) -> str:
    return (
        f"{SOURCE_KEY}[Appendix B.1, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, {row['Raw_Lect_Code']}]"
    )


def read_snapshot() -> list[dict[str, str]]:
    if hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() != SNAPSHOT_SHA256:
        raise AssertionError("frozen Pahari/Pothwari snapshot fingerprint drift")
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 3472:
        raise AssertionError(f"source cell-count drift: {len(rows)}")
    return rows


def build() -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    source_rows = read_snapshot()
    installed: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    for row in source_rows:
        target = row["Source_Scope"] == "target"
        response = row["Status"] == "response"
        install = target and response
        did = dialect_id(row["Lect_Code"]) if target else ""
        source = citation(row)
        entry_key = ""
        if install:
            entry_key = (
                f"{ENTRY_PREFIX}:g{int(row['Item']):03d}:"
                f"{row['Lect_Code'].lower()}"
            )
            notes = (
                f"ESR 2010-012 {row['Site']} list; source reliability {row['Reliability']}; "
                "source transcription uses the report's Indological Phonetic Script"
            )
            if row["Excluded_From_Similarity"] == "Yes":
                notes += "; source marks this prompt excluded from lexical-similarity calculations"
            if row["Review"]:
                notes += "; " + row["Review"]
            installed.append({
                "Language_ID": LANGUAGE_ID,
                "Parameter_ID": "",
                "Form": row["Form"],
                "Gloss": row["Gloss"],
                "Native": "",
                "Phonemic": row["Form"],
                "Notes": notes,
                "Source": source,
                "Cognateset": "",
                "Etymology": "",
                "Entry_Key": entry_key,
                "Variant_Of_Key": "",
                "Borrowed_From_Key": "",
                "Derivation_Parent_Keys": "",
                "Tags": dialect_tag(row["Lect_Code"], row["Site"]),
            })

        if install:
            status, reason = "installed", ""
        elif row["Source_Scope"] == "hindko_control":
            status = "excluded"
            reason = "Abbottabad/Mansehra Hindko comparison list retained audit-only"
            if not response:
                reason += "; source cell is blank"
        else:
            status = "excluded"
            reason = "target cell is blank"
        audit.append({
            "Record_Type": "wordlist cell",
            "Source_Key": SOURCE_KEY,
            "PDF_Page": row["PDF_Page"],
            "Printed_Page": row["Printed_Page"],
            "Item": row["Item"],
            "Gloss": row["Gloss"],
            "Excluded_From_Similarity": row["Excluded_From_Similarity"],
            "Raw_Lect_Code": row["Raw_Lect_Code"],
            "Lect_Code": row["Lect_Code"],
            "Site": row["Site"],
            "District": row["District"],
            "Reliability": row["Reliability"],
            "Raw_Form": row["Raw_Form"],
            "Form": row["Form"],
            "Review": row["Review"],
            "Source_Scope": row["Source_Scope"],
            "Source_Status": row["Status"],
            "Status": status,
            "Reason": reason,
            "Language_ID": LANGUAGE_ID if target else "",
            "Dialect_ID": did,
            "Citation": source,
            "Entry_Key": entry_key,
        })

    if len(installed) != 3038 or len(audit) != 3472:
        raise AssertionError(
            f"installation topology drift: installed={len(installed)} audit={len(audit)}"
        )
    if Counter(row["Status"] for row in audit) != Counter({"installed": 3038, "excluded": 434}):
        raise AssertionError("audit status topology drift")
    if Counter(row["Lect_Code"] for row in audit if row["Status"] == "installed") != Counter({
        code: 217 for code in (
            "MOS", "GHO", "DEW", "AYU", "KOH", "NIL", "THA", "LOR",
            "OSI", "MUZ", "DUN", "BHA", "MIR", "GUJ",
        )
    }):
        raise AssertionError("target list topology drift")
    if len({row["Entry_Key"] for row in installed}) != len(installed):
        raise AssertionError("duplicate installed Entry_Key")
    if any(not row["Form"] or row["Form"] != row["Phonemic"] for row in installed):
        raise AssertionError("blank or mismatched installed transcription")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in installed):
        raise AssertionError("non-NFC installed form")
    if any(row["Parameter_ID"] or row["Cognateset"] or row["Etymology"] for row in installed):
        raise AssertionError("phonetic similarity must not become an etymological claim")

    manifest = {
        "source_key": SOURCE_KEY,
        "publisher_file": "silesr2010-012.pdf",
        "publisher_file_sha256": SOURCE_SHA256,
        "publisher_pdf_url": (
            "https://www.sil.org/system/files/reapdata/51/43/60/"
            "51436058131979368214107509780343634713/silesr2010_012.pdf"
        ),
        "publisher_pdf_archive_capture": (
            "https://web.archive.org/web/20140907143711id_/http://www.sil.org/system/"
            "files/reapdata/51/43/60/51436058131979368214107509780343634713/"
            "silesr2010_012.pdf"
        ),
        "source_extent": "262 PDF pages; Appendix B.1 printed pp. 147-202 (PDF pp. 153-208)",
        "snapshot_file": str(SNAPSHOT.relative_to(REPO)),
        "snapshot_sha256": SNAPSHOT_SHA256,
        "source_prompts": 217,
        "source_lists": 16,
        "source_cells": len(source_rows),
        "source_response_cells": 3454,
        "source_blank_cells": 18,
        "target_lists": 14,
        "target_response_cells": 3038,
        "installed_responses": len(installed),
        "hindko_control_cells": 434,
        "hindko_control_responses": 416,
        "source_similarity_excluded_prompts": 11,
        "printed_AUS_codes_normalized_to_OSI": 14,
        "audit_records": len(audit),
        "audit_status_counts": dict(Counter(row["Status"] for row in audit)),
        "replacement_or_private_use_glyphs": 0,
        "unparsed_cells": 0,
        "ocr_used": False,
        "etymology_edges": 0,
        "coordinate_policy": (
            "the report supplies regional maps but no point coordinates; target dialect "
            "coordinates remain blank rather than receiving invented centroids"
        ),
    }
    return installed, audit, manifest


def write() -> None:
    installed, audit, manifest = build()
    with INSTALLED.open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=FORM_FIELDS).writerows(installed)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"installed={len(installed)} excluded={len(audit) - len(installed)} audit={len(audit)}")


if __name__ == "__main__":
    write()
