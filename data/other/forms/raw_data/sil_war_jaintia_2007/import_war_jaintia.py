#!/usr/bin/env python3
"""Install the seven War-Jaintia wordlists in SIL ESR 2007-013."""

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
INSTALLED = FORMS / "20260828-sil-war-jaintia.csv"
AUDIT = RAW_ROOT / "20260828-sil-war-jaintia-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-war-jaintia-manifest.json"

SOURCE_KEY = "brightbill-kim-kim2007warjaintia"
KEY_PREFIX = "silwarjaintia2007"
PDF_SHA256 = "df28fa5fb8961c2b5029428cace0567d5aa2bb078112903d237517c25521657e"
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
    "A": ("Niralapunji", "sil-war-jaintia-2007-niralapunji"),
    "B": ("Aliachora", "sil-war-jaintia-2007-aliachora"),
    "C": ("Dabolchora", "sil-war-jaintia-2007-dabolchora"),
    "D": ("Singur", "sil-war-jaintia-2007-singur"),
    "E": ("Barenga", "sil-war-jaintia-2007-barenga"),
    "I": ("Magurchora", "sil-war-jaintia-2007-magurchora"),
    "J": ("Amlarem", "sil-war-jaintia-2007-amlarem"),
}
CONTROLS = {
    "F": ("Noksia", "Synteng/Pnar comparison"),
    "G": ("Jaintiapur", "Jowai/Pnar comparison"),
    "H": ("Panai", "Lyngngam comparison"),
    "K": ("Shella", "Khasi War comparison"),
    "L": ("Shillong", "standard Khasi comparison"),
}


def dialect_tag(code: str) -> str:
    site, dialect_id = SITES[code]
    return f"dialect:WarJaintia:{quote(dialect_id, safe='')}:{quote(site, safe='')}"


def locator(row: dict[str, str], site: str) -> str:
    return (
        f"{SOURCE_KEY}[Appendix B.3, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, {site}]"
    )


def build() -> tuple[list[dict], list[dict], dict]:
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(source_rows) != 1690:
        raise AssertionError(f"printed response-count drift: {len(source_rows)}")

    installed = []
    audit = []
    site_responses = defaultdict(int)
    for row in source_rows:
        for code in row["Site_Codes"]:
            site_responses[int(row["Item"]), code] += 1
            response = site_responses[int(row["Item"]), code]
            target = code in SITES
            if target:
                site, dialect_id = SITES[code]
                group_name = "War-Jaintia target wordlist"
                status, reason = "installed", ""
            elif code in CONTROLS:
                site, group_name = CONTROLS[code]
                dialect_id = ""
                status, reason = "excluded", f"{group_name}; not War-Jaintia"
            else:
                site, dialect_id, group_name = "Undefined site U", "", "source anomaly"
                status = "excluded"
                reason = "undefined code U printed at item 119; Appendix B.2 defines only A-L"
            source = locator(row, site)
            entry_key = ""
            if target:
                entry_key = f"{KEY_PREFIX}:g{int(row['Item']):03d}:{dialect_id}:i{response}"
                installed.append({
                    "Language_ID": "WarJaintia", "Parameter_ID": "",
                    "Form": row["Form"], "Gloss": row["Gloss"], "Native": "",
                    "Phonemic": row["Form"],
                    "Notes": (
                        f"Appendix B.3 lexical-similarity group {row['Similarity_Group']}; "
                        "War-Jaintia survey site"
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
                "Review": row["Review"], "Uncertainty": "" if code != "U" else "printed site-code error",
                "Status": status, "Reason": reason,
                "Language_ID": "WarJaintia" if target else "",
                "Dialect_ID": dialect_id, "Source": source, "Entry_Key": entry_key,
            })

    if len(installed) != 2030 or len(audit) != 3459:
        raise AssertionError(f"topology drift: installed={len(installed)} audit={len(audit)}")
    if len({row["Entry_Key"] for row in installed}) != len(installed):
        raise AssertionError("duplicate installed entry key")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in installed):
        raise AssertionError("non-NFC installed form")
    manifest = {
        "source_key": SOURCE_KEY,
        "publisher_series_page": "https://www.silbangladesh.org/resources/publications/survay_report",
        "preserved_official_pdf_url": (
            "https://web.archive.org/web/20151001194723id_/"
            "http://www-01.sil.org/silesr/2007/silesr2007-013.pdf"
        ),
        "publisher_pdf_sha256": PDF_SHA256,
        "publisher_pdf_bytes": 2051039, "publisher_pdf_pages": 153,
        "included_scope": "Appendix B.3, printed pp. 57-87, all seven War-Jaintia site lists",
        "excluded_scope": (
            "Pnar, Lyngngam, Khasi War and standard Khasi comparison lists F-H/K-L; "
            "one undefined printed site code U"
        ),
        "transcription_file": str(TRANSCRIPTION.relative_to(REPO)),
        "transcription_sha256": hashlib.sha256(TRANSCRIPTION.read_bytes()).hexdigest(),
        "source_items": 307, "printed_response_records": len(source_rows),
        "expanded_attestations": len(audit), "target_installed": len(installed),
        "control_records": 1428, "undefined_site_records": 1,
        "audit_records": len(audit),
        "audit_status_counts": dict(Counter(row["Status"] for row in audit)),
        "legacy_encoding": "SIL-SAG-IPA / SAGIPA2Uni.map v1.0 (2007)",
        "legacy_pua_symbols": 17, "legacy_pua_occurrences": 2398,
        "unparsed_lines": 0, "unmapped_legacy_symbols": 0,
        "empty_prompt_items": [
            21, 30, 36, 39, 40, 41, 42, 51, 60, 64, 65, 67, 72, 75, 88,
            124, 146, 149, 159, 168, 171, 194, 199, 203, 209, 239, 240, 255,
            257, 301, 306,
        ],
        "printed_anomalies": {
            "item_119_site_U": "excluded because Appendix B.2 defines only site codes A-L",
            "item_137_group_A": "retained as the printed tenth similarity group after 1-9",
        },
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
    print(f"installed={len(installed)} controls=1428 undefined=1 audit={len(audit)}")


if __name__ == "__main__":
    write()
