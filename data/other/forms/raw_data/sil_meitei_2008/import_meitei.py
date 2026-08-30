#!/usr/bin/env python3
"""Install the eight Meitei wordlists in SIL ESR 2008-002."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
FORMS = REPO / "data/other/forms"
RAW_ROOT = FORMS / "raw_data"
TRANSCRIPTION = HERE / "wordlists.tsv"
INSTALLED = FORMS / "20260828-sil-meitei.csv"
AUDIT = RAW_ROOT / "20260828-sil-meitei-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-meitei-manifest.json"

SOURCE_KEY = "kim-kim2008meitei"
KEY_PREFIX = "silmeitei2008"
PDF_SHA256 = "d86fcbb4da2124da0a3ba6a7b48a7c63288fbe45e73d9d7bd03dc83e5e0b4d47"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Report", "Printed_Page", "Gloss_Number", "Gloss",
    "Site_Code", "Site_Name", "Dialect_Group", "Response", "Similarity_Group",
    "Raw_Transcription", "Transcription", "Review", "Uncertainty", "Status",
    "Reason", "Language_ID", "Dialect_ID", "Source", "Entry_Key",
]
SITES = {
    "1": ("Mukabil", "sil-meitei-2008-mukabil", "Pangal community, Bangladesh"),
    "2": ("Humerjan", "sil-meitei-2008-humerjan", "Meitei community, Bangladesh"),
    "3": ("Shivganj", "sil-meitei-2008-shivganj", "Meitei community, Bangladesh"),
    "4": ("Shivnagar", "sil-meitei-2008-shivnagar", "Meitei community, Bangladesh"),
    "5": ("Choto Dhamai", "sil-meitei-2008-choto-dhamai", "Meitei community, Bangladesh"),
    "6": ("Kunagaon", "sil-meitei-2008-kunagaon", "Meitei community, Bangladesh"),
    "7": ("Lilong Bazaar", "sil-meitei-2008-lilong-bazaar", "Manipur comparison variety"),
    "8": ("Imphal", "sil-meitei-2008-imphal", "Manipur comparison variety"),
}


def dialect_tag(code: str) -> str:
    site, dialect_id, _ = SITES[code]
    return f"dialect:Manipuri:{quote(dialect_id, safe='')}:{quote(site, safe='')}"


def locator(row: dict[str, str], site: str) -> str:
    return (
        f"{SOURCE_KEY}[Appendix B.3, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, {site}]"
    )


def build() -> tuple[list[dict], list[dict], dict]:
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(source_rows) != 1219:
        raise AssertionError(f"printed response-count drift: {len(source_rows)}")

    installed = []
    audit = []
    site_responses = defaultdict(int)
    for row in source_rows:
        for code in row["Site_Codes"]:
            target = code in SITES
            if target:
                site, dialect_id, group_name = SITES[code]
            else:
                site, dialect_id, group_name = "Standard Bangla", "", "comparison control"
            site_responses[int(row["Item"]), code] += 1
            response = site_responses[int(row["Item"]), code]
            source = locator(row, site)
            entry_key = ""
            status = "installed" if target else "excluded"
            reason = "" if target else "standard Bangla comparison list; not Meitei"
            if target:
                entry_key = f"{KEY_PREFIX}:g{int(row['Item']):03d}:{dialect_id}:i{response}"
                installed.append({
                    "Language_ID": "Manipuri", "Parameter_ID": "",
                    "Form": row["Form"], "Gloss": row["Gloss"], "Native": "",
                    "Phonemic": row["Form"],
                    "Notes": (
                        f"Appendix B.3 lexical-similarity group {row['Similarity_Group']}; "
                        f"{group_name}"
                    ),
                    "Source": source, "Cognateset": "", "Etymology": "",
                    "Entry_Key": entry_key, "Variant_Of_Key": "",
                    "Borrowed_From_Key": "", "Derivation_Parent_Keys": "",
                    "Tags": dialect_tag(code),
                })
            audit.append({
                "Record_Type": "expanded wordlist attestation", "Report": SOURCE_KEY,
                "Printed_Page": row["Printed_Page"], "Gloss_Number": row["Item"],
                "Gloss": row["Gloss"], "Site_Code": code, "Site_Name": site,
                "Dialect_Group": group_name, "Response": response,
                "Similarity_Group": row["Similarity_Group"],
                "Raw_Transcription": row["Raw_Form"], "Transcription": row["Form"],
                "Review": row["Review"], "Uncertainty": "", "Status": status,
                "Reason": reason, "Language_ID": "Manipuri" if target else "",
                "Dialect_ID": dialect_id, "Source": source, "Entry_Key": entry_key,
            })

    if len(installed) != 2406 or len(audit) != 2713:
        raise AssertionError(f"topology drift: installed={len(installed)} audit={len(audit)}")
    if len({row["Entry_Key"] for row in installed}) != len(installed):
        raise AssertionError("duplicate installed entry key")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in installed):
        raise AssertionError("non-NFC installed form")
    manifest = {
        "source_key": SOURCE_KEY,
        "publisher_archive_record": "https://www.sil.org/resources/archives/9145",
        "publisher_pdf_url": (
            "https://www.sil.org/system/files/reapdata/84/82/92/"
            "84829231524183376942587881497312279207/silesr2008_002.pdf"
        ),
        "publisher_pdf_sha256": PDF_SHA256,
        "publisher_pdf_bytes": 916584,
        "publisher_pdf_pages": 126,
        "included_scope": "Appendix B.3, printed pp. 45-68, all eight Meitei site lists",
        "excluded_scope": "Standard Dhaka Bangla comparison list (site code 0)",
        "transcription_file": str(TRANSCRIPTION.relative_to(REPO)),
        "transcription_sha256": hashlib.sha256(TRANSCRIPTION.read_bytes()).hexdigest(),
        "source_items": 307, "printed_response_records": len(source_rows),
        "expanded_attestations": len(audit), "target_installed": len(installed),
        "control_records": 307, "audit_records": len(audit),
        "audit_status_counts": dict(Counter(row["Status"] for row in audit)),
        "legacy_encoding": "SIL-SAG-IPA / SAGIPA2Uni.map v1.0 (2007)",
        "legacy_pua_symbols": 25, "legacy_pua_occurrences": 2534,
        "unparsed_lines": 0, "unmapped_legacy_symbols": 0,
        "etymology_edges": 0,
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
    print(f"installed={len(installed)} controls=307 audit={len(audit)}")


if __name__ == "__main__":
    write()
