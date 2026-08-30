#!/usr/bin/env python3
"""Install the 36 Appendix B wordlists from SSNP volume 4."""

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
INSTALLED = FORMS / "20260828-ssnp04.csv"
AUDIT = RAW_ROOT / "20260828-ssnp04-audit.csv"
MANIFEST = RAW_ROOT / "20260828-ssnp04-manifest.json"

SOURCE_KEY = "hallberg1992pashto"
SOURCE_SHA256 = "83e2d833c06ecb4e40bfb0d316061d6b398b743bac299dc870c90c88a4b96f18"
SNAPSHOT_SHA256 = "e3be47472daf107beea441ed6d823bb07f1460d1c4ae5adb687e2773a1db8e15"
ENTRY_PREFIX = "ssnp04"

# code: printed label, language ID, glottocode, source group, reliability,
# latitude, longitude, coordinate/locality note
LIST_META = {
    "PES": ("Peshawar Pashto", "Psht", "nort2646", "Northern settled", "A", 34.0151, 71.5249, "Peshawar city"),
    "CHS": ("Charsadda Pashto", "Psht", "nort2646", "Northern settled", "A", 34.1453, 71.7308, "Charsadda city"),
    "MAR": ("Mardan Pashto", "Psht", "nort2646", "Northern settled", "A", 34.1989, 72.0402, "Mardan city"),
    "SWA": ("Swabi Pashto", "Psht", "nort2646", "Northern settled", "A", 34.1202, 72.4698, "Swabi city"),
    "MAD": ("Madyan Pashto", "Psht", "nort2646", "Northern settled", "A", 35.1460, 72.5360, "Madyan, Swat"),
    "MIN": ("Saidu Sharif/Mingora Pashto", "Psht", "nort2646", "Northern settled", "A", 34.7717, 72.3600, "Mingora/Saidu Sharif urban area"),
    "BAT": ("Batagram Pashto", "Psht", "nort2646", "Northern settled", "A", 34.6796, 73.0280, "Battagram city"),
    "BAF": ("Baffa Pashto", "Psht", "nort2646", "Northern settled", "A", 34.4377, 73.2237, "Baffa, Mansehra District"),
    "OGI": ("Oghi Pashto", "Psht", "nort2646", "Northern settled", "B", 34.5110, 73.2710, "Oghi, Mansehra District"),
    "DIR": ("Dir Pashto", "Psht", "nort2646", "Northern settled", "B", 35.2058, 71.8756, "Dir city"),
    "BAJ": ("Bajaur Pashto", "Psht", "nort2646", "Northern tribal", "B", 34.7460, 71.5240, "Khar/Bajaur regional point"),
    "MOH": ("Mohmand Pashto", "Psht", "nort2646", "Northern tribal", "B", 34.3220, 71.4140, "Ghalanai/Mohmand regional point"),
    "NIG": ("Ningrahar Pashto", "Psht", "nort2646", "Northern tribal", "A", 34.4340, 70.4470, "Jalalabad/Nangarhar regional point, Afghanistan"),
    "SHN": ("Shinwari Pashto", "Psht", "nort2646", "Northern tribal", "A", 34.1030, 71.1410, "Landi Kotal/Shinwari regional point"),
    "BAR": ("Bar/Loi Shilman Pashto", "Psht", "nort2646", "Northern tribal", "B", 34.1650, 71.0500, "Shilman Valley regional point"),
    "MAL": ("Mallagori Pashto", "Psht", "nort2646", "Northern tribal", "B", 34.0640, 71.3000, "Mullagori regional point"),
    "ZKH": ("Zakha Khel Afridi Pashto", "Psht", "nort2646", "Northern tribal", "B", 33.9990, 71.1000, "Zakha Khel/Afridi regional point"),
    "JAM": ("Jamrud Afridi Pashto", "Psht", "nort2646", "Northern tribal", "A", 34.0030, 71.3810, "Jamrud"),
    "TIR": ("Tirah Afridi Pashto", "Psht", "nort2646", "Northern tribal", "B", 33.9000, 70.5000, "Tirah Valley regional point"),
    "JAL": ("Jallozai Pashto", "Psht", "nort2646", "Northern", "B", 33.9990, 71.7550, "Jallozai"),
    "CHE": ("Cherat Pashto", "Psht", "nort2646", "Northern", "B", 33.8240, 71.8920, "Cherat"),
    "PAR": ("Parachinar Pashto", "Psht", "", "Middle settled/tribal", "B", 33.8990, 70.1010, "Parachinar"),
    "HAN": ("Hangu Pashto", "Psht", "", "Middle settled/tribal", "B", 33.5280, 71.0570, "Hangu"),
    "TAL": ("Thal Pashto", "Psht", "", "Middle settled/tribal", "B", 33.3590, 70.5420, "Thall, Hangu District"),
    "KRK": ("Karak Pashto", "Psht", "cent1973", "Central", "", 33.1160, 71.0950, "Karak; source prints no reliability code"),
    "LAK": ("Lakki Marwat Pashto", "Psht", "cent1973", "Central", "B", 32.6079, 70.9110, "Lakki Marwat"),
    "BAN": ("Bannu Pashto", "Psht", "cent1973", "Central", "B", 32.9854, 70.6027, "Bannu"),
    "MIR": ("Miran Shah Pashto", "Psht", "cent1973", "Central", "B", 33.0006, 70.0712, "Miran Shah, North Waziristan"),
    "WAA": ("Wana Pashto", "Psht", "cent1973", "Central", "B", 32.2980, 69.5720, "Wana, South Waziristan"),
    "QUE": ("Quetta Pashto", "Psht", "sout2649", "Southern", "A", 30.1798, 66.9750, "Quetta"),
    "CHA": ("Chaman Pashto", "Psht", "sout2649", "Southern", "A", 30.9236, 66.4512, "Chaman"),
    "PAS": ("Pishin Pashto", "Psht", "sout2649", "Southern", "A", 30.5830, 66.9960, "Pishin"),
    "KAK": ("Pishin Kakari Pashto", "Psht", "sout2649", "Southern", "A", 30.5830, 66.9960, "Pishin; Kakari speaker/list"),
    "KHR": ("Kandahar Pashto", "Psht", "sout2649", "Southern", "A", 31.6200, 65.7160, "Kandahar, Afghanistan"),
    "WCI": ("Harnai Waneci", "wne", "wane1241", "Waneci", "A", 30.1000, 67.9380, "Speakers from Harnai recorded while living in Quetta"),
    "ORM": ("Kaniguram Ormuri", "Orm", "ormu1247", "Ormuri", "A", 32.4800, 69.7500, "Kaniguram, South Waziristan"),
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Source_Key", "PDF_Page", "Printed_Page", "Item", "Gloss",
    "List_Code", "List_Name", "Source_Group", "Reliability", "Raw_Form", "Form",
    "Continuation_Lines", "Review", "Status", "Reason", "Language_ID",
    "Dialect_ID", "Citation", "Entry_Key",
]


def dialect_id(code: str) -> str:
    return f"ssnp04-{code.lower()}"


def dialect_tag(code: str) -> str:
    name, language_id, *_ = LIST_META[code]
    return f"dialect:{language_id}:{dialect_id(code)}:{quote(name, safe='')}"


def locator(row: dict[str, str]) -> str:
    return (
        f"{SOURCE_KEY}[Appendix B, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, {row['List_Code']}]"
    )


def read_snapshot() -> list[dict[str, str]]:
    if hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() != SNAPSHOT_SHA256:
        raise AssertionError("frozen SSNP volume 4 snapshot fingerprint drift")
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 7200:
        raise AssertionError(f"source cell-count drift: {len(rows)}")
    return rows


def build() -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    source_rows = read_snapshot()
    if {row["List_Code"] for row in source_rows} != set(LIST_META):
        raise AssertionError("list metadata topology drift")

    installed = []
    audit = []
    for row in source_rows:
        code = row["List_Code"]
        name, language_id, _, group, reliability, _, _, _ = LIST_META[code]
        install = row["Status"] == "response"
        citation = locator(row)
        entry_key = f"{ENTRY_PREFIX}:g{int(row['Item']):03d}:{code.lower()}" if install else ""
        if install:
            reliability_note = f"source reliability {reliability}" if reliability else "source prints no reliability code"
            installed.append({
                "Language_ID": language_id,
                "Parameter_ID": "",
                "Form": row["Form"],
                "Gloss": row["Gloss"],
                "Native": "",
                "Phonemic": row["Form"],
                "Notes": (
                    f"SSNP volume 4 {group} list; {reliability_note}; source field "
                    "transcription is explicitly not a full phonological analysis"
                ),
                "Source": citation,
                "Cognateset": "",
                "Etymology": "",
                "Entry_Key": entry_key,
                "Variant_Of_Key": "",
                "Borrowed_From_Key": "",
                "Derivation_Parent_Keys": "",
                "Tags": dialect_tag(code),
            })

        if install:
            audit_status, reason = "installed", ""
        elif row["Status"] == "no_entry":
            audit_status, reason = "excluded", "source prints -- rather than a lexical response"
        else:
            audit_status, reason = "excluded", "source cell is blank"
        audit.append({
            "Record_Type": "wordlist cell",
            "Source_Key": SOURCE_KEY,
            "PDF_Page": row["PDF_Page"],
            "Printed_Page": row["Printed_Page"],
            "Item": row["Item"],
            "Gloss": row["Gloss"],
            "List_Code": code,
            "List_Name": name,
            "Source_Group": group,
            "Reliability": reliability,
            "Raw_Form": row["Raw_Form"],
            "Form": row["Form"],
            "Continuation_Lines": row["Continuation_Lines"],
            "Review": row["Review"],
            "Status": audit_status,
            "Reason": reason,
            "Language_ID": language_id,
            "Dialect_ID": dialect_id(code),
            "Citation": citation,
            "Entry_Key": entry_key,
        })

    if len(installed) != 7131 or len(audit) != 7200:
        raise AssertionError("installation topology drift")
    if Counter(row["Language_ID"] for row in installed) != Counter({
        "Psht": 6732, "wne": 199, "Orm": 200,
    }):
        raise AssertionError("installed language-count drift")
    if len({row["Entry_Key"] for row in installed}) != len(installed):
        raise AssertionError("duplicate installed Entry_Key")
    if any(not row["Form"] or row["Form"] != row["Phonemic"] for row in installed):
        raise AssertionError("blank or mismatched installed transcription")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in installed):
        raise AssertionError("non-NFC installed transcription")
    if any(row["Parameter_ID"] or row["Cognateset"] for row in installed):
        raise AssertionError("lexical-similarity lists must remain unetymologised")

    manifest = {
        "source_key": SOURCE_KEY,
        "publisher_file": "32847_SSNP04.pdf",
        "publisher_file_sha256": SOURCE_SHA256,
        "publisher_catalog_url": (
            "https://web.archive.org/web/20121022060337id_/"
            "http://www.sil.org/sociolx/pubs/abstract.asp?id=32847"
        ),
        "publisher_pdf_url": "http://www.sil.org/sociolx/pubs/32847_SSNP04.pdf",
        "publisher_pdf_archive_capture": (
            "https://web.archive.org/web/20121011032301id_/"
            "http://www.sil.org/sociolx/pubs/32847_ssnp04.pdf"
        ),
        "source_extent": "194 PDF pages; Appendix B printed pp. 79-146 (PDF pp. 97-164)",
        "snapshot_file": str(SNAPSHOT.relative_to(REPO)),
        "snapshot_sha256": SNAPSHOT_SHA256,
        "source_numbered_prompts_printed": 200,
        "source_missing_prompt_numbers": [24, 29, 32, 50, 173, 174, 175, 176, 195, 208],
        "source_lists": 36,
        "source_pashto_lists": 34,
        "source_cells": len(source_rows),
        "installed_responses": len(installed),
        "printed_no_entry_cells": 68,
        "blank_cells": 1,
        "audit_records": len(audit),
        "installed_language_counts": dict(Counter(row["Language_ID"] for row in installed)),
        "visual_continuation_cells": sum(int(row["Continuation_Lines"]) > 0 for row in source_rows),
        "visual_continuation_lines": sum(int(row["Continuation_Lines"]) for row in source_rows),
        "unparsed_cells": 0,
        "replacement_or_private_use_glyphs": 0,
        "ocr_used": False,
        "legacy_font": "SILDoulosNP",
        "etymology_edges": 0,
        "coordinate_policy": (
            "modern locality points or explicitly labelled regional points; source survey "
            "labels and speaker origins are authoritative, coordinates are not source coordinates"
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
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"installed={len(installed)} excluded={len(audit) - len(installed)} audit={len(audit)}")


if __name__ == "__main__":
    write()
