#!/usr/bin/env python3
"""Install the ten Bangladesh Kuki-Chin wordlists in SIL ESR 2011-025."""

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
INSTALLED = FORMS / "20260828-sil-kuki-chin-bangladesh.csv"
AUDIT = RAW_ROOT / "20260828-sil-kuki-chin-bangladesh-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-kuki-chin-bangladesh-manifest.json"

SOURCE_KEY = "kim-roy-sangma2011kukichin"
KEY_PREFIX = "silkukichin2011"
PDF_SHA256 = "d0506535e6040bafebe88a3f3db5217f68ffd42179651ef9316bcbcb2272b230"
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
    "a": ("Bilaichari", "Pangkhua", "sil-kuki-chin-2011-bilaichari-pangkhua"),
    "l": ("Konglak", "Pangkhua", "sil-kuki-chin-2011-konglak"),
    "i": ("Bethel Para (Bawm)", "BawmChin", "sil-kuki-chin-2011-bethel-para-bawm"),
    "b": ("Jamunachari", "BawmChin", "sil-kuki-chin-2011-jamunachari"),
    "j": ("Bethel Para (Mizo)", "Mizo", "sil-kuki-chin-2011-bethel-para-mizo"),
    "m": ("Mahmuam Para", "Mizo", "sil-kuki-chin-2011-mahmuam-para"),
    "c": ("Boro Kukyachari", "AshoChin", "sil-kuki-chin-2011-boro-kukyachari"),
    "k": ("Ghungurumukh Para", "AshoChin", "sil-kuki-chin-2011-ghungurumukh-para"),
    "g": ("Manglung Headman Para", "KhumiChin", "sil-kuki-chin-2011-manglung-headman-para"),
    "h": ("Prongphung Para", "KhumiChin", "sil-kuki-chin-2011-prongphung-para"),
}
CONTROLS = {
    "0": ("Standard Bangla", "standard Bangla comparison"),
    "e": ("Myanmar Khumi", "Myanmar Khumi comparison supplied by another linguist"),
}


def dialect_tag(code: str) -> str:
    site, language_id, dialect_id = SITES[code]
    return f"dialect:{language_id}:{quote(dialect_id, safe='')}:{quote(site, safe='')}"


def locator(row: dict[str, str], site: str) -> str:
    return (
        f"{SOURCE_KEY}[Appendix A.3, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, {site}]"
    )


def build() -> tuple[list[dict], list[dict], dict]:
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(source_rows) != 2565:
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
                site, language_id, dialect_id = SITES[code]
                group_name = "Bangladesh Kuki-Chin target wordlist"
                status, reason = "installed", ""
            else:
                site, group_name = CONTROLS[code]
                language_id = dialect_id = ""
                status, reason = "excluded", f"{group_name}; outside target scope"
                if not row["Form"]:
                    reason += "; printed no entry"
            source = locator(row, site)
            entry_key = ""
            if target:
                entry_key = f"{KEY_PREFIX}:g{int(row['Item']):03d}:{dialect_id}:i{response}"
                installed.append({
                    "Language_ID": language_id, "Parameter_ID": "",
                    "Form": row["Form"], "Gloss": row["Gloss"], "Native": "",
                    "Phonemic": row["Form"],
                    "Notes": (
                        f"Appendix A.3 lexical-similarity group {row['Similarity_Group']}; "
                        f"source label {site}; Bangladesh Kuki-Chin survey site"
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
                "Review": row["Review"], "Uncertainty": "",
                "Status": status, "Reason": reason, "Language_ID": language_id,
                "Dialect_ID": dialect_id, "Source": source, "Entry_Key": entry_key,
            })

    if len(installed) != 3235 or len(audit) != 3875:
        raise AssertionError(f"topology drift: installed={len(installed)} audit={len(audit)}")
    if len({row["Entry_Key"] for row in installed}) != len(installed):
        raise AssertionError("duplicate installed entry key")
    if any(not row["Form"] for row in installed):
        raise AssertionError("empty target form")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in installed):
        raise AssertionError("non-NFC installed form")
    language_counts = Counter(row["Language_ID"] for row in installed)
    expected_languages = Counter({
        "AshoChin": 648, "KhumiChin": 648, "Pangkhua": 647,
        "BawmChin": 647, "Mizo": 645,
    })
    if language_counts != expected_languages:
        raise AssertionError(f"language-count drift: {language_counts}")

    manifest = {
        "source_key": SOURCE_KEY,
        "publisher_archive_url": "https://www.silbangladesh.org/resources/archives/41669",
        "publisher_series_page": "https://www.silbangladesh.org/resources/publications/survay_report",
        "public_appendix_mirror": "Google Drive file 0BxA1OYBm_BU0M3Q2M3lwYk1fVmc",
        "publisher_pdf_sha256": PDF_SHA256,
        "publisher_pdf_bytes": 1834732,
        "publisher_pdf_pages": 127,
        "included_scope": (
            "Appendix A.3, printed pp. 50-88, all ten Bangladesh Kuki-Chin site lists"
        ),
        "excluded_scope": (
            "standard Bangla control 0 and external Myanmar Khumi comparison e; "
            "all non-lexical appendices"
        ),
        "transcription_file": str(TRANSCRIPTION.relative_to(REPO)),
        "transcription_sha256": hashlib.sha256(TRANSCRIPTION.read_bytes()).hexdigest(),
        "source_items": 306,
        "printed_response_records": len(source_rows),
        "expanded_attestations": len(audit),
        "target_installed": len(installed),
        "language_counts": dict(language_counts),
        "standard_bangla_controls": 307,
        "myanmar_khumi_controls": 333,
        "printed_no_entry_records": 53,
        "audit_records": len(audit),
        "audit_status_counts": dict(Counter(row["Status"] for row in audit)),
        "legacy_encoding": "subsetted SAG-IPA-SILManuscript / SAGIPA2Uni.map v1.0",
        "legacy_pua_symbols": 65,
        "legacy_pua_occurrences": 16029,
        "unparsed_lines": 0,
        "unmapped_legacy_symbols": 0,
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
    print("installed=3235 bangla_controls=307 myanmar_khumi_controls=333 audit=3875")


if __name__ == "__main__":
    write()
