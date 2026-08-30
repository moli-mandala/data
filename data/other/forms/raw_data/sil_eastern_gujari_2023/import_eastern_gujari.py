#!/usr/bin/env python3
"""Install the eight visually reviewed Indian Eastern Gujari wordlists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[4]
REVIEWED = HERE / "reviewed_transcription.tsv"
OUTPUT = DATA_ROOT / "data/other/forms/20260828-sil-eastern-gujari.csv"
AUDIT = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-eastern-gujari-audit.csv"
MANIFEST = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-eastern-gujari-manifest.json"
SOURCE_KEY = "hugoniot-polster-ahmad-rajan2023easterngujari"
PDF_SHA256 = "41352b2db97dbd059a1bc229a8ed370fed700c1726f3886a580cba586137475e"
PDF_URL = "https://www.sil.org/system/files/reapdata/16/64/68/166468818346814241493507732958257420275/JLSR2023_002.pdf"
ARCHIVE_URL = "https://www.sil.org/resources/archives/95899"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Source_Key", "PDF_Page", "Printed_Page", "Page_Line", "Item", "Gloss",
    "Source_Code", "List", "Role", "Existing_SSNP_Dialect", "Source_Cell",
    "Verified_Cell", "Similarity_Groups", "Expanded_Forms", "Record_Type",
    "Review_Method", "Review_Status", "Confidence", "Status", "Reason",
    "Language_ID", "Dialect_ID", "Citation", "Installed_Count", "Entry_Keys",
]

# code, role, dialect id, display, historical source locality
LISTS = {
    "Urdu": ("URD", "Urdu control", "", "", "Pakistan survey control"),
    "Chitral/Pak": ("CHT", "republished SSNP list", "SSNP-gojri-CHT", "", "Ashriki, Shishi Koh valley"),
    "Settlet Swat/Pak": ("SSW", "republished SSNP list", "SSNP-gojri-SSW", "", "Peshmal, Swat valley"),
    "Gilgit/Pak": ("GLT", "republished SSNP list", "SSNP-gojri-GLT", "", "Naltar Bala"),
    "Kaghan/Pak": ("KGH", "republished SSNP list", "SSNP-gojri-KGH", "", "Mittikot above Balakot"),
    "North. Azad/Pak": ("NAK", "republished SSNP list", "SSNP-gojri-NAK", "", "Muzaffarabad / Subri"),
    "Centr. Azad/Pak": ("CAK", "republished SSNP list", "SSNP-gojri-CAK", "", "Rawalakot / Trarkhel"),
    "Udhampur/J&K": ("UDH", "new Indian target", "sil-eastern-gujari-1996-udhampur", "Udhampur", "Udhampur district, Jammu and Kashmir"),
    "Jammu/J&K": ("JAM", "new Indian target", "sil-eastern-gujari-1996-jammu", "Jammu", "Jammu district, Jammu and Kashmir"),
    "Chamba/H.P.": ("CHA", "new Indian target", "sil-eastern-gujari-1996-chamba", "Chamba", "Chamba district, Himachal Pradesh"),
    "Rampur/H.P.": ("RAM", "new Indian target", "sil-eastern-gujari-1996-rampur", "Rampur", "Shimla district, Himachal Pradesh"),
    "Nalagarh/H.P.": ("NAL", "new Indian target", "sil-eastern-gujari-1996-nalagarh", "Nalagarh", "Solan district, Himachal Pradesh"),
    "Dehradun/U.P.": ("DEH", "new Indian target", "sil-eastern-gujari-1996-dehra-dun", "Dehra Dun", "Dehra Dun district, Uttar Pradesh (source-era label)"),
    "Kotdwara/U.P.": ("KOT", "new Indian target", "sil-eastern-gujari-1996-kotdwara", "Kotdwara", "Uttar Pradesh (district not supplied)"),
    "Haldwani/U.P.": ("HAL", "new Indian target", "sil-eastern-gujari-1996-haldwani", "Haldwani", "Naini Tal district, Uttar Pradesh (source-era label)"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag(dialect_id: str, display: str) -> str:
    return f"dialect:Goj:{quote(dialect_id, safe='')}:{quote(display, safe='')}"


def parse_cell(cell: str) -> list[tuple[str, str]]:
    if re.search(r"\b0\s+no entry\b", cell, re.I):
        return []
    alternatives = re.split(r"\s*/\s*(?=\d+\s)", cell.strip())
    parsed: list[tuple[str, str]] = []
    for alternative in alternatives:
        matched = re.fullmatch(r"(\d+)\s+(.+)", alternative.strip())
        if not matched:
            raise ValueError(f"unparsed source response: {cell!r}")
        parsed.append((matched.group(1), unicodedata.normalize("NFC", matched.group(2).strip())))
    return parsed


def load_reviewed() -> list[dict[str, str]]:
    with REVIEWED.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 3_150 or len({(r["Item"], r["List"]) for r in rows}) != 3_150:
        raise ValueError("expected the complete 210 x 15 review ledger")
    if any(r["Review_Status"] != "complete" or r["Confidence"] != "high" for r in rows):
        raise ValueError("every cell must have complete high-confidence visual review")
    if any(r["Source_Cell"] != r["Verified_Cell"] for r in rows):
        raise ValueError("review corrections must be explicit before import")
    if any(r["List"] not in LISTS for r in rows):
        raise ValueError("unknown list in review ledger")
    return rows


def build(rows: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]], dict[str, int]]:
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    duplicate_alternatives = 0
    for row in rows:
        code, role, dialect_id, display, _ = LISTS[row["List"]]
        target = role == "new Indian target"
        alternatives = parse_cell(row["Verified_Cell"])
        citation = f"{SOURCE_KEY}[Appendix B, PDF p. {row['PDF_Page']}, printed p. {row['Printed_Page']}, item {row['Item']}, list {code}]"
        entry_keys: list[str] = []
        seen_forms: set[str] = set()
        groups: list[str] = []
        expanded: list[str] = []
        for alt_number, (group, form) in enumerate(alternatives, 1):
            groups.append(group)
            expanded.append(form)
            if not target:
                continue
            if form in seen_forms:
                duplicate_alternatives += 1
                continue
            seen_forms.add(form)
            entry_key = f"sileasterngujari2023:p{int(row['PDF_Page']):03d}:i{int(row['Item']):03d}:{code}:a{alt_number}"
            entry_keys.append(entry_key)
            forms.append([
                "Goj", "", form, row["Gloss"], "", form,
                f"source lexical-similarity group {group}; source list {row['List']}",
                citation, "", "", entry_key, "", "", "", tag(dialect_id, display),
            ])
        if row["Record_Type"] == "blank":
            status, reason = "excluded", "source explicitly prints no entry"
        elif target:
            status, reason = "installed", ("one exact repeated alternative installed once" if len(entry_keys) < len(alternatives) else "")
        elif role == "republished SSNP list":
            status, reason = "excluded", f"audit-only reprint; primary SSNP data already installed as {dialect_id}"
        else:
            status, reason = "excluded", "Urdu comparison control"
        audit.append(dict(zip(AUDIT_FIELDS, [
            SOURCE_KEY, row["PDF_Page"], row["Printed_Page"], row["Page_Line"],
            row["Item"], row["Gloss"], code, row["List"], role,
            dialect_id if role == "republished SSNP list" else "", row["Source_Cell"],
            row["Verified_Cell"], " | ".join(groups), " | ".join(expanded),
            row["Record_Type"],
            "manual visual comparison against 180-dpi rendered canonical PDF page; Unicode text layer used only as extraction scaffold",
            row["Review_Status"], row["Confidence"], status, reason,
            "Goj" if status == "installed" else "", dialect_id, citation,
            str(len(entry_keys)), " | ".join(entry_keys),
        ])))

    target_blanks = sum(r["Record_Type"] == "blank" and LISTS[r["List"]][1] == "new Indian target" for r in rows)
    non_target_blanks = sum(r["Record_Type"] == "blank" and LISTS[r["List"]][1] != "new Indian target" for r in rows)
    counts = {
        "prompts": 210,
        "printed_lists": 15,
        "conceptual_source_cells_manually_reviewed": 3150,
        "target_conceptual_cells": 1680,
        "republished_ssnp_conceptual_cells": 1260,
        "urdu_control_conceptual_cells": 210,
        "attested_cells": sum(r["Record_Type"] == "response" for r in rows),
        "confirmed_blank_cells": target_blanks + non_target_blanks,
        "confirmed_target_blank_cells": target_blanks,
        "confirmed_non_target_blank_cells": non_target_blanks,
        "excluded_attested_republished_ssnp_cells": sum(r["Record_Type"] == "response" and LISTS[r["List"]][1] == "republished SSNP list" for r in rows),
        "excluded_attested_urdu_control_cells": sum(r["Record_Type"] == "response" and r["List"] == "Urdu" for r in rows),
        "target_attested_cells": sum(r["Record_Type"] == "response" and LISTS[r["List"]][1] == "new Indian target" for r in rows),
        "target_printed_alternative_occurrences": sum(len(parse_cell(r["Verified_Cell"])) for r in rows if LISTS[r["List"]][1] == "new Indian target"),
        "duplicate_target_alternatives_audit_only": duplicate_alternatives,
        "installed_forms": len(forms),
        "audit_rows": len(audit),
        "ambiguous_or_illegible_cells": 0,
        "unresolved_transcriptions": 0,
    }
    return forms, audit, counts


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows = load_reviewed()
    forms, audit, counts = build(rows)
    if args.install:
        write_csv(OUTPUT, forms)
        with AUDIT.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
            writer.writeheader(); writer.writerows(audit)
        manifest = {
            "source_key": SOURCE_KEY,
            "title": "A Sociolinguistic Profile of Eastern Gujari",
            "authors": ["Ken Hugoniot", "Dietmar Polster", "Bashir Ahmad", "Kennedy Rajan"],
            "year": 2023,
            "fieldwork_year": 1996,
            "series": "Journal of Language Survey Reports 2023-002",
            "archive_url": ARCHIVE_URL, "pdf_url": PDF_URL,
            "pdf_sha256": PDF_SHA256, "pdf_size_bytes": 9_149_165, "pdf_pages": 121,
            "scope": "Appendix B, PDF pp. 41-76; lexical matrix PDF pp. 42-76, printed pp. 34-68",
            "review": {
                "authority": "rendered canonical PDF pages",
                "text_layer": "extraction scaffold only; all 3,150 printed cells visually compared",
                "ocr": "not used; appendix is born-digital Unicode",
                "image_only_or_handwritten_cells": 0,
                "unresolved": [],
            },
            "counts": counts,
            "artifacts": {
                "reviewed_transcription": {"path": str(REVIEWED.relative_to(DATA_ROOT)), "sha256": file_sha256(REVIEWED)},
                "installed": {"path": str(OUTPUT.relative_to(DATA_ROOT)), "sha256": file_sha256(OUTPUT)},
                "audit": {"path": str(AUDIT.relative_to(DATA_ROOT)), "sha256": file_sha256(AUDIT)},
            },
        }
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
