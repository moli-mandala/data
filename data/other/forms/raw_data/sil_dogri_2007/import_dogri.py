#!/usr/bin/env python3
"""Install the Batote Dogri wordlist from SIL ESR 2007-017."""

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
INSTALLED = FORMS / "20260828-sil-dogri.csv"
AUDIT = RAW_ROOT / "20260828-sil-dogri-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-dogri-manifest.json"

SOURCE_KEY = "brightbill-turner2007dogri"
SOURCE_PDF = "silesr2007_017.pdf"
SOURCE_SHA256 = "04fa21ccf3ca7317ef1a1b3e587b4f1c058b3fb773ea56724d726945a12622c0"
OFFICIAL_MAP_SHA256 = "f2bb1070e8393f83e6ea83d8b08ee0b07e23bbe0176ccb9ca97b76793809df31"
DIALECT_ID = "sil-dogri-2007-batote"
ENTRY_PREFIX = "sildogri2007"
BLANK_ITEMS = {11: "breast", 23: "urine", 24: "feces"}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Source_Key", "PDF_Page", "Printed_Page", "Item", "Gloss",
    "Site", "Raw_Form", "Form", "Review", "Uncertainty", "Status", "Reason",
    "Language_ID", "Dialect_ID", "Citation", "Entry_Key",
]


def dialect_tag() -> str:
    return f"dialect:dog:{quote(DIALECT_ID, safe='')}:Batote"


def locator(row: dict[str, str]) -> str:
    return (
        f"{SOURCE_KEY}[Appendix B, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, Batote]"
    )


def build() -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(source_rows) != 210:
        raise AssertionError(f"source item-count drift: {len(source_rows)}")

    installed: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    for row in source_rows:
        item = int(row["Item"])
        citation = locator(row)
        form = row["Form"]
        status = "installed" if form else "excluded"
        reason = "" if form else "source prints a blank response cell"
        entry_key = ""
        if form:
            entry_key = f"{ENTRY_PREFIX}:g{item:03d}:{DIALECT_ID}:i1"
            installed.append({
                "Language_ID": "dog",
                "Parameter_ID": "",
                "Form": form,
                "Gloss": row["Gloss"],
                "Native": "",
                "Phonemic": form,
                "Notes": "Appendix B Batote wordlist; elicited 11 April 2005",
                "Source": citation,
                "Cognateset": "",
                "Etymology": "",
                "Entry_Key": entry_key,
                "Variant_Of_Key": "",
                "Borrowed_From_Key": "",
                "Derivation_Parent_Keys": "",
                "Tags": dialect_tag(),
            })
        audit.append({
            "Record_Type": "wordlist prompt",
            "Source_Key": SOURCE_KEY,
            "PDF_Page": row["PDF_Page"],
            "Printed_Page": row["Printed_Page"],
            "Item": row["Item"],
            "Gloss": row["Gloss"],
            "Site": "Batote",
            "Raw_Form": row["Raw_Form"],
            "Form": form,
            "Review": row["Review"],
            "Uncertainty": "",
            "Status": status,
            "Reason": reason,
            "Language_ID": "dog",
            "Dialect_ID": DIALECT_ID,
            "Citation": citation,
            "Entry_Key": entry_key,
        })

    if len(installed) != 207 or len(audit) != 210:
        raise AssertionError(
            f"topology drift: installed={len(installed)} audit={len(audit)}"
        )
    excluded = {int(row["Item"]): row["Gloss"] for row in audit if row["Status"] == "excluded"}
    if excluded != BLANK_ITEMS:
        raise AssertionError(f"blank-item drift: {excluded}")
    if len({row["Entry_Key"] for row in installed}) != len(installed):
        raise AssertionError("duplicate installed Entry_Key")
    if any(not row["Form"] or row["Form"] != row["Phonemic"] for row in installed):
        raise AssertionError("blank or mismatched installed transcription")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in installed):
        raise AssertionError("non-NFC installed form")

    manifest = {
        "source_key": SOURCE_KEY,
        "publisher_file": SOURCE_PDF,
        "publisher_file_sha256": SOURCE_SHA256,
        "publisher_archive_record": "https://www.sil.org/resources/archives/9015",
        "publisher_pdf_url": (
            "https://www.sil.org/system/files/reapdata/60/68/92/"
            "60689248968204682975758522816050115406/silesr2007_017.pdf"
        ),
        "source_extent": "29 pages; Appendix B printed pp. 26-28",
        "source_created": "field research May 2004; Batote follow-up elicited 11 April 2005",
        "official_converter": "SIL-IPA93-2001.map v14",
        "official_converter_map_sha256": OFFICIAL_MAP_SHA256,
        "snapshot_file": str(SNAPSHOT.relative_to(REPO)),
        "snapshot_sha256": hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        "source_prompts": len(source_rows),
        "installed_responses": len(installed),
        "blank_prompt_items": sorted(BLANK_ITEMS),
        "audit_records": len(audit),
        "audit_status_counts": dict(Counter(row["Status"] for row in audit)),
        "target_sites": 1,
        "comparison_wordlists_reported_but_not_published": 5,
        "unparsed_lines": 0,
        "unmapped_legacy_symbols": 0,
        "ocr_used": False,
        "etymology_edges": 0,
    }
    return installed, audit, manifest


def write() -> None:
    installed, audit, manifest = build()
    with INSTALLED.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FORM_FIELDS)
        writer.writerows(installed)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"installed={len(installed)} blanks={len(audit) - len(installed)} audit={len(audit)}")


if __name__ == "__main__":
    write()
