#!/usr/bin/env python3
"""Install the six Bishnupriya village wordlists from SIL ESR 2008-003."""

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
INSTALLED = FORMS / "20260828-sil-bishnupriya.csv"
AUDIT = RAW_ROOT / "20260828-sil-bishnupriya-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-bishnupriya-manifest.json"

SOURCE_KEY = "kim-kim2008bishnupriya"
KEY_PREFIX = "silbishnupriya2008"
TRANSCRIPT_SHA256 = "1c42bad6a4ee278b4056397f3c5db960b091d5e1df749ff021137a0398aeac8b"
EMPTY_ITEMS = {194, 218, 221, 222, 258, 259, 301, 303, 306}
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Report", "Printed_Page", "Column", "Gloss_Number", "Gloss",
    "Site_Code", "Site_Name", "Dialect_Group", "Response", "Similarity_Group",
    "Raw_Transcription", "Transcription", "Review", "Uncertainty", "Status", "Reason",
    "Language_ID", "Dialect_ID", "Source", "Entry_Key",
]
SITES = {
    "a": ("Tilakpur", "sil-bishnupriya-2008-tilakpur", "Madoi Gang"),
    "b": ("Soi Sri", "sil-bishnupriya-2008-soi-sri", "Rajar Gang"),
    "c": ("Gulerhaor", "sil-bishnupriya-2008-gulerhaor", "Rajar Gang"),
    "d": ("Dhonitila", "sil-bishnupriya-2008-dhonitila", "Rajar Gang"),
    "e": ("Machimpur", "sil-bishnupriya-2008-machimpur", "Rajar Gang"),
    "f": ("Madhapur", "sil-bishnupriya-2008-madhapur", "Madoi Gang"),
}


def dialect_tag(code: str) -> str:
    site, dialect_id, _ = SITES[code]
    return (
        f"dialect:Bishnupriya:{quote(dialect_id, safe='')}:"
        f"{quote(site, safe='')}"
    )


def locator(row: dict[str, str], site: str) -> str:
    return (
        f"{SOURCE_KEY}[Appendix B.3, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, {site}]"
    )


def build() -> tuple[list[dict], list[dict], dict]:
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(source_rows) != 746:
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
            reason = "" if target else "standard Bangla comparison list; not Bishnupriya"
            if target:
                entry_key = (
                    f"{KEY_PREFIX}:g{int(row['Item']):03d}:{dialect_id}:i{response}"
                )
                notes = (
                    f"Appendix B.3 lexical-similarity group {row['Similarity_Group']}; "
                    f"{group_name} village"
                )
                installed.append({
                    "Language_ID": "Bishnupriya", "Parameter_ID": "",
                    "Form": row["Form"], "Gloss": row["Gloss"], "Native": "",
                    "Phonemic": row["Form"], "Notes": notes, "Source": source,
                    "Cognateset": "", "Etymology": "", "Entry_Key": entry_key,
                    "Variant_Of_Key": "", "Borrowed_From_Key": "",
                    "Derivation_Parent_Keys": "", "Tags": dialect_tag(code),
                })
            uncertainty = ""
            if "lowercase o" in row["Review"]:
                uncertainty = "source site code is visibly lowercase o; interpreted as Bangla 0"
            audit.append({
                "Record_Type": "expanded wordlist attestation", "Report": SOURCE_KEY,
                "Printed_Page": row["Printed_Page"], "Column": row["Column"],
                "Gloss_Number": row["Item"], "Gloss": row["Gloss"],
                "Site_Code": code, "Site_Name": site, "Dialect_Group": group_name,
                "Response": response, "Similarity_Group": row["Similarity_Group"],
                "Raw_Transcription": row["Raw_Form"], "Transcription": row["Form"],
                "Review": row["Review"], "Uncertainty": uncertainty,
                "Status": status, "Reason": reason,
                "Language_ID": "Bishnupriya" if target else "",
                "Dialect_ID": dialect_id, "Source": source, "Entry_Key": entry_key,
            })

    # These headings occur in the 1--307 sequence but the source prints no response beneath them.
    # They are source-level gaps, not linguistic forms and not invented "no entry" attestations.
    by_item = {int(row["Item"]): row for row in source_rows}
    glosses = {
        194: "to tell", 218: "to lie, fib", 221: "to kill", 222: "to die",
        258: "hungry", 259: "thirsty", 301: "2s (honorific)",
        303: "3s (female)", 306: "2p (honorific)",
    }
    pages = {194: 46, 218: 47, 221: 47, 222: 47, 258: 49, 259: 49,
             301: 52, 303: 52, 306: 52}
    for item in sorted(EMPTY_ITEMS):
        audit.append({
            "Record_Type": "empty prompt", "Report": SOURCE_KEY,
            "Printed_Page": pages[item], "Column": 1 if item == 194 else 2,
            "Gloss_Number": item, "Gloss": glosses[item], "Site_Code": "",
            "Site_Name": "", "Dialect_Group": "all lists", "Response": "",
            "Similarity_Group": "", "Raw_Transcription": "", "Transcription": "",
            "Review": "heading verified on the pinned source-page image", "Uncertainty": "",
            "Status": "excluded", "Reason": "source prints the prompt heading with no responses",
            "Language_ID": "", "Dialect_ID": "",
            "Source": f"{SOURCE_KEY}[Appendix B.3, printed p. {pages[item]}, item {item}]",
            "Entry_Key": "",
        })

    if by_item.keys() & EMPTY_ITEMS:
        raise AssertionError("empty prompt unexpectedly has a printed response")
    if len(installed) != 1801 or len(audit) != 2108:
        raise AssertionError(
            f"topology drift: installed={len(installed)} audit={len(audit)}"
        )
    if len({row["Entry_Key"] for row in installed}) != len(installed):
        raise AssertionError("duplicate installed entry key")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in installed):
        raise AssertionError("non-NFC installed form")

    manifest = {
        "source_key": SOURCE_KEY,
        "publisher_archive_record": "https://www.sil.org/resources/archives/9100",
        "publisher_pdf_url": (
            "https://www.sil.org/system/files/reapdata/74/05/96/"
            "74059695388257055076108658933060858639/silesr2008_003.pdf"
        ),
        "publisher_pdf_sha256": None,
        "publisher_pdf_note": (
            "The official SIL PDF was visually verified in-browser; its endpoint returned a "
            "Cloudflare HTML challenge to the command-line downloader, so no PDF digest is claimed."
        ),
        "extraction_scaffold": (
            "Public Slideshare copy 2624796: fixed-layout __NEXT_DATA__ transcript and "
            "2048-pixel source-page rasters; SIL remains the bibliographic authority."
        ),
        "wordlist_transcript_sha256": TRANSCRIPT_SHA256,
        "transcription_file": str(TRANSCRIPTION.relative_to(REPO)),
        "transcription_sha256": hashlib.sha256(TRANSCRIPTION.read_bytes()).hexdigest(),
        "source_items": 307, "printed_response_records": len(source_rows),
        "expanded_attestations": 2099, "target_installed": len(installed),
        "control_records": 298, "empty_prompts": sorted(EMPTY_ITEMS),
        "audit_records": len(audit),
        "audit_status_counts": dict(Counter(row["Status"] for row in audit)),
        "legacy_pua_symbols": 14, "legacy_pua_occurrences": 947,
        "superscript_aspiration_markers": 161,
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
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"installed={len(installed)} controls={manifest['control_records']} "
        f"empty_prompts={len(EMPTY_ITEMS)} audit={len(audit)}"
    )


if __name__ == "__main__":
    write()
