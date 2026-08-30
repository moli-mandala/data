#!/usr/bin/env python3
"""Install the seven Jaunsari wordlists in SIL ESR 2008-013."""

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
TRANSCRIPTION = HERE / "wordlists.tsv"
INSTALLED = FORMS / "20260828-sil-jaunsari.csv"
AUDIT = RAW_ROOT / "20260828-sil-jaunsari-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-jaunsari-manifest.json"

SOURCE_KEY = "john2008jaunsari"
SOURCE_PDF = "silesr2008_013.pdf"
SOURCE_SHA256 = "e6b3b6d54c061d03614b27618f0f06d2138f07c47dc1a266d45b0fe16bd75f68"
OFFICIAL_MAP_SHA256 = "a989926e91d4b562df20758cbb613f0177fce33d1c2e9e02195087e94f1f2930"
KEY_PREFIX = "siljaunsari2008"
DISQUALIFIED = {11: "breast", 23: "urine", 24: "feces"}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Report", "PDF_Page", "Printed_Page", "Gloss_Number", "Gloss",
    "Site_Code", "Site_Name", "Source_Label", "Comparison_Role", "Response",
    "Similarity_Group", "Transcription", "Review", "Uncertainty", "Status", "Reason",
    "Language_ID", "Dialect_ID", "Source", "Entry_Key",
]

TARGETS = {
    "A": ("Chakrata", "sil-jaunsari-2008-chakrata", "previous Garhwali survey list"),
    "B": ("Bhandroli", "sil-jaunsari-2008-bhandroli", "Jaunsari-Bawari"),
    "C": ("Chapnu", "sil-jaunsari-2008-chapnu", "Jaunsari"),
    "D": ("Khanaad", "sil-jaunsari-2008-khanaad", "Jaunsari"),
    "K": ("Korwa", "sil-jaunsari-2008-korwa", "Jaunsari"),
    "L": ("Lakhamandal", "sil-jaunsari-2008-lakhamandal", "Jaunsari-Garhwali border mix"),
    "M": ("Maindrath", "sil-jaunsari-2008-maindrath", "Jaunsari-Bawari"),
}


def dialect_tag(code: str) -> str:
    site, dialect_id, _ = TARGETS[code]
    return f"dialect:jaun:{quote(dialect_id, safe='')}:{quote(site, safe='')}"


def locator(row: dict[str, str]) -> str:
    return (
        f"{SOURCE_KEY}[Appendix A.2, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, {row['Site_Name']}]"
    )


def build() -> tuple[list[dict], list[dict], dict]:
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(source_rows) != 2729:
        raise AssertionError(f"source response-count drift: {len(source_rows)}")

    installed = []
    audit = []
    for row in source_rows:
        code = row["Source_Code"]
        target = code in TARGETS
        status = "installed" if target else "excluded"
        reason = "" if target else "comparison-language control; not a Jaunsari target list"
        dialect_id = TARGETS[code][1] if target else ""
        source = locator(row)
        entry_key = ""
        if target:
            site, _, variety = TARGETS[code]
            entry_key = (
                f"{KEY_PREFIX}:g{int(row['Item']):03d}:{dialect_id}:"
                f"i{int(row['Response'])}"
            )
            notes = f"Appendix A.2 lexical-similarity group {row['Similarity_Group']}; {variety}"
            installed.append({
                "Language_ID": "jaun", "Parameter_ID": "", "Form": row["Form"],
                "Gloss": row["Gloss"], "Native": "", "Phonemic": row["Form"],
                "Notes": notes, "Source": source, "Cognateset": "", "Etymology": "",
                "Entry_Key": entry_key, "Variant_Of_Key": "", "Borrowed_From_Key": "",
                "Derivation_Parent_Keys": "", "Tags": dialect_tag(code),
            })
        audit.append({
            "Record_Type": "wordlist response", "Report": SOURCE_KEY,
            "PDF_Page": row["PDF_Page"], "Printed_Page": row["Printed_Page"],
            "Gloss_Number": row["Item"], "Gloss": row["Gloss"], "Site_Code": code,
            "Site_Name": row["Site_Name"], "Source_Label": row["Source_Label"],
            "Comparison_Role": row["Role"], "Response": row["Response"],
            "Similarity_Group": row["Similarity_Group"], "Transcription": row["Form"],
            "Review": "decoded from the SAG-IPA text layer with SIL's official converter map",
            "Uncertainty": "", "Status": status, "Reason": reason,
            "Language_ID": "jaun" if target else "", "Dialect_ID": dialect_id,
            "Source": source, "Entry_Key": entry_key,
        })

    for item, gloss in DISQUALIFIED.items():
        source = f"{SOURCE_KEY}[Appendix A.2, printed p. 36, item {item}]"
        audit.append({
            "Record_Type": "item exclusion", "Report": SOURCE_KEY, "PDF_Page": 36,
            "Printed_Page": 36, "Gloss_Number": item, "Gloss": gloss, "Site_Code": "",
            "Site_Name": "", "Source_Label": "", "Comparison_Role": "all lists",
            "Response": "", "Similarity_Group": "", "Transcription": "",
            "Review": "classified from the source note", "Uncertainty": "",
            "Status": "excluded", "Reason": "source says the gloss was disqualified and removed",
            "Language_ID": "", "Dialect_ID": "", "Source": source, "Entry_Key": "",
        })

    if len(installed) != 1619 or len(audit) != 2732:
        raise AssertionError(
            f"topology drift: installed={len(installed)} audit={len(audit)}"
        )
    if len({row["Entry_Key"] for row in installed}) != len(installed):
        raise AssertionError("duplicate installed entry key")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in installed):
        raise AssertionError("non-NFC installed form")

    manifest = {
        "source_key": SOURCE_KEY,
        "publisher_file": SOURCE_PDF,
        "publisher_file_sha256": SOURCE_SHA256,
        "publisher_archive_record": "https://www.sil.org/resources/archives/9074",
        "publisher_pdf_url": (
            "https://www.sil.org/system/files/reapdata/13/61/38/"
            "136138879857923730999543684874534219059/silesr2008_013.pdf"
        ),
        "official_converter": "SIL-SAG-IPA / SAGIPA2Uni.map v1.0 (2007)",
        "official_converter_map_sha256": OFFICIAL_MAP_SHA256,
        "transcription_file": str(TRANSCRIPTION.relative_to(REPO)),
        "transcription_sha256": hashlib.sha256(TRANSCRIPTION.read_bytes()).hexdigest(),
        "items_in_source_list": 210,
        "printed_items": 207,
        "disqualified_items": sorted(DISQUALIFIED),
        "source_response_records": len(source_rows),
        "target_installed": len(installed),
        "control_records": sum(row["Role"] != "target" for row in source_rows),
        "audit_records": len(audit),
        "audit_status_counts": dict(Counter(row["Status"] for row in audit)),
        "unparsed_lines": 0,
        "unmapped_legacy_symbols": 0,
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
    print(
        f"installed={len(installed)} controls={manifest['control_records']} "
        f"audit={len(audit)}"
    )


if __name__ == "__main__":
    write()
