#!/usr/bin/env python3
"""Install the six Haryanvi wordlists in JLSR 2024-011.

The appendix is an image-only scan.  ``manual_transcription.tsv`` is the
authoritative, cell-by-cell human transcription.  Tesseract output is retained
only as structural evidence for the four excluded comparison lists and must
never supply an installed form.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
FORMS = REPO / "data/other/forms"
RAW_ROOT = FORMS / "raw_data"
MANUAL = HERE / "manual_transcription.tsv"
OCR_SCAFFOLD = HERE / "transcription.tsv"
INSTALLED = FORMS / "20260828-sil-haryanvi.csv"
AUDIT = RAW_ROOT / "20260828-sil-haryanvi-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-haryanvi-manifest.json"

SOURCE_KEY = "webster2024haryanvi"
SOURCE_SHA256 = "53121a1b9803ba502092866080e3bdb35457bc6040dcc7f47da508eca1fef2e2"
MANUAL_SHA256 = "9d3b065452db45cc240e085497988795471facdfa48aa592d21c1734b5dd47fe"
OCR_SCAFFOLD_SHA256 = "04daeadcd74f45dd0715ea9ef3813f7185e10086cd6fd0922a23d73e11ee926a"
LANGUAGE_ID = "kaithal"
ENTRY_PREFIX = "silharyanvi2024"

SITE_ORDER = ("HRT", "HJN", "HFT", "HNG", "BPL", "HTR", "HLH", "PBG", "HIN", "PUN")
TARGET_SITES = ("HRT", "HJN", "HFT", "HNG", "HTR", "HLH")
COMPARISON_SITES = ("BPL", "PBG", "HIN", "PUN")
SITE_METADATA = {
    "HRT": {
        "name": "Rohtak Haryanvi/Bangru",
        "location": "Rohtak, Rohtak district, Haryana",
        "verification": "double-checked: yes",
    },
    "HJN": {
        "name": "Jind Haryanvi",
        "location": "Jind, Jind district, Haryana",
        "verification": "Appendix A.2 metadata omitted",
    },
    "HFT": {
        "name": "Fatehabad Haryanvi/Bagri",
        "location": "Fatehabad, Hissar district, Haryana",
        "verification": "double-checked: no",
    },
    "HNG": {
        "name": "Dehar (Narayangarh) Haryanvi",
        "location": "Dehar, Narayangarh Tehsil, Ambala district, Haryana",
        "verification": "double-checked: partially",
    },
    "HTR": {
        "name": "Taoru Haryanvi/Mewati",
        "location": "Taoru, Gurgaon district, Haryana",
        "verification": "Appendix A.2 metadata omitted",
    },
    "HLH": {
        "name": "Loharu Haryanvi/Bagri",
        "location": "Loharu, Mahendragarh district, Haryana",
        "verification": "Appendix A.2 metadata omitted",
    },
    "BPL": {
        "name": "Palwal Braj Bhasha/Haryanvi comparison",
        "location": "Aalahpur, Palwal Tehsil, Faridabad district, Haryana",
        "verification": "elicited in a group",
    },
    "PBG": {
        "name": "Kuthard Baghati Pahari comparison",
        "location": "Kuthard, Kasali Tehsil, Solan district, Himachal Pradesh",
        "verification": "double-check status unknown",
    },
    "HIN": {
        "name": "Standard North Indian Hindustani comparison",
        "location": "North India; many sources",
        "verification": "not stated",
    },
    "PUN": {
        "name": "Standard Punjabi comparison",
        "location": "Fatehgarh/Churian, Guridaspur district, Punjab",
        "verification": "double-checked: yes",
    },
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Source_Key", "PDF_Page", "Printed_Page", "Column", "Item",
    "Gloss", "Site", "Role", "Source_Name", "Location", "Source_Raw",
    "Manual_Transcription", "Resolved_Forms", "Similarity_Groups", "Qualifiers",
    "Manual_Review", "Item_Review_Note", "Uncertainty_Type", "OCR_Evidence_Primary",
    "OCR_Evidence_Secondary", "OCR_Evidence_Latin", "Source_Status", "Status",
    "Reason", "Language_ID", "Dialect_ID", "Citation", "Entry_Keys",
]

GROUP = re.compile(r"^(\d+(?:[/,]\d+)*)\s+(.+)$")
QUALIFIER = re.compile(r"^(.*?)\s+\(([^()]*)\)$")
BRACKET_XREF = re.compile(r"^\[same as item (\d+)\]$", re.IGNORECASE)
PAREN_XREF = re.compile(r"^\(same as item (\d+)\)$", re.IGNORECASE)
SITE_CODE = re.compile(r"\b(?:HRT|HJN|HFT|HNG|HTR|HLH)\b")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dialect_id(site: str) -> str:
    return f"sil-haryanvi-2024-{site.lower()}"


def dialect_tag(site: str) -> str:
    metadata = SITE_METADATA[site]
    return (
        f"dialect:{LANGUAGE_ID}:{quote(dialect_id(site), safe='')}:"
        f"{quote(metadata['name'], safe='')}"
    )


def citation(row: dict[str, str], site: str) -> str:
    return (
        f"{SOURCE_KEY}[Appendix A.3, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, {site}]"
    )


def read_inputs() -> tuple[list[dict[str, str]], dict[tuple[int, str], dict[str, str]]]:
    if sha256(MANUAL) != MANUAL_SHA256:
        raise AssertionError("frozen manual Haryanvi transcription fingerprint drift")
    if sha256(OCR_SCAFFOLD) != OCR_SCAFFOLD_SHA256:
        raise AssertionError("frozen Haryanvi OCR scaffold fingerprint drift")

    with MANUAL.open(encoding="utf-8", newline="") as stream:
        manual = list(csv.DictReader(stream, delimiter="\t"))
    if len(manual) != 210 or [int(row["Item"]) for row in manual] != list(range(1, 211)):
        raise AssertionError("manual ledger must contain items 1-210 exactly once and in order")
    if any(row["Review"] != "manual-scan" for row in manual):
        raise AssertionError("every target row must retain its manual-scan review marker")

    with OCR_SCAFFOLD.open(encoding="utf-8", newline="") as stream:
        scaffold_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(scaffold_rows) != 2100:
        raise AssertionError("OCR scaffold must account for 210 prompts x 10 lists")
    scaffold = {(int(row["Item"]), row["Site"]): row for row in scaffold_rows}
    if set(scaffold) != {(item, site) for item in range(1, 211) for site in SITE_ORDER}:
        raise AssertionError("OCR scaffold prompt/list topology drift")

    for row in manual:
        for site in TARGET_SITES:
            evidence = scaffold[(int(row["Item"]), site)]
            if (
                evidence["PDF_Page"] != row["PDF_Page"]
                or evidence["Printed_Page"] != row["Printed_Page"]
                or evidence["Column"] != row["Column"]
            ):
                raise AssertionError(f"manual/OCR locator drift at item {row['Item']} {site}")
    return manual, scaffold


def cross_reference(raw: str) -> int | None:
    match = BRACKET_XREF.match(raw) or PAREN_XREF.match(raw)
    return int(match.group(1)) if match else None


def parse_direct(raw: str) -> list[dict[str, str]]:
    """Expand one manually transcribed response cell into printed alternatives."""
    if raw == "[blank]" or raw.startswith("[elicitation note:") or cross_reference(raw):
        return []
    variants: list[dict[str, str]] = []
    previous_group = ""
    for part in re.split(r"\s+/\s*", raw):
        match = GROUP.match(part)
        if match:
            group, form = match.groups()
            previous_group = group
        elif previous_group:
            group, form = previous_group, part
        else:
            raise AssertionError(f"response alternative lacks a similarity group: {raw!r}")
        qualifier = ""
        qualified = QUALIFIER.match(form)
        if qualified:
            form, qualifier = qualified.groups()
        form = unicodedata.normalize("NFC", form.strip())
        if not form:
            raise AssertionError(f"empty form parsed from {raw!r}")
        variants.append({"group": group, "form": form, "qualifier": qualifier})
    return variants


def cell_review_note(row: dict[str, str], site: str) -> str:
    note = row["Uncertainty"].strip()
    mentioned = set(SITE_CODE.findall(note))
    return note if note and (not mentioned or site in mentioned) else ""


def uncertainty_type(note: str) -> str:
    lowered = note.lower()
    if any(word in lowered for word in ("faint", "clipped", "unresolved")):
        return "transcription"
    if any(word in lowered for word in ("unlabelled", "structurally", "source repeats")):
        return "source-layout"
    return ""


def source_status(raw: str) -> str:
    if raw == "[blank]":
        return "blank"
    if raw.startswith("[elicitation note:"):
        return "elicitation-note"
    if cross_reference(raw):
        return "cross-reference"
    return "response"


def build() -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    manual, scaffold = read_inputs()
    by_item = {int(row["Item"]): row for row in manual}

    def resolve(item: int, site: str, stack: tuple[int, ...] = ()) -> tuple[list[dict[str, str]], int | None]:
        if item in stack:
            raise AssertionError(f"cyclic Haryanvi cross-reference at item {item} {site}")
        raw = by_item[item][site]
        referenced = cross_reference(raw)
        if referenced is None:
            return parse_direct(raw), None
        if referenced not in by_item:
            raise AssertionError(f"cross-reference to absent item {referenced}")
        variants, _ = resolve(referenced, site, stack + (item,))
        if not variants:
            raise AssertionError(f"cross-reference resolves to no form at item {item} {site}")
        return [dict(variant) for variant in variants], referenced

    installed: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    direct_variants = 0
    direct_response_cells = 0
    cross_reference_cells = 0
    target_blank_cells = 0
    target_elicitation_cells = 0

    for row in manual:
        item = int(row["Item"])
        for site in SITE_ORDER:
            evidence = scaffold[(item, site)]
            metadata = SITE_METADATA[site]
            cite = citation(row, site)
            if site in TARGET_SITES:
                raw = row[site]
                status = source_status(raw)
                forms, referenced = resolve(item, site)
                if status == "response":
                    direct_response_cells += 1
                    direct_variants += len(forms)
                elif status == "cross-reference":
                    cross_reference_cells += 1
                elif status == "blank":
                    target_blank_cells += 1
                elif status == "elicitation-note":
                    target_elicitation_cells += 1

                review_note = cell_review_note(row, site)
                review_type = uncertainty_type(review_note)
                entry_keys: list[str] = []
                for variant_index, variant in enumerate(forms, 1):
                    entry_key = (
                        f"{ENTRY_PREFIX}:i{item:03d}:{site.lower()}:v{variant_index:02d}"
                    )
                    entry_keys.append(entry_key)
                    notes = [
                        f"Appendix A.3 lexical-similarity group {variant['group']}",
                        "manually transcribed from the image-only source scan",
                    ]
                    if variant["qualifier"]:
                        notes.append(f"source qualifier: {variant['qualifier']}")
                    if referenced is not None:
                        notes.append(f"source cross-reference to item {referenced}")
                    tags = [dialect_tag(site)]
                    if review_type:
                        tags.append("uncertain")
                    installed.append({
                        "Language_ID": LANGUAGE_ID,
                        "Parameter_ID": "",
                        "Form": variant["form"],
                        "Gloss": row["Gloss"],
                        "Native": "",
                        "Phonemic": variant["form"],
                        "Notes": "; ".join(notes),
                        "Source": cite,
                        "Cognateset": "",
                        "Etymology": "",
                        "Entry_Key": entry_key,
                        "Variant_Of_Key": "",
                        "Borrowed_From_Key": "",
                        "Derivation_Parent_Keys": "",
                        "Tags": " ".join(tags),
                    })

                if status in {"response", "cross-reference"}:
                    install_status = "installed"
                    reason = (
                        f"source cross-reference resolved to item {referenced} in the same list"
                        if referenced is not None else ""
                    )
                elif status == "blank":
                    install_status, reason = "excluded", "source cell is visibly blank"
                else:
                    install_status = "excluded"
                    reason = "source gives an elicitation instruction, not a lexical response"
                audit.append({
                    "Record_Type": "wordlist cell",
                    "Source_Key": SOURCE_KEY,
                    "PDF_Page": row["PDF_Page"],
                    "Printed_Page": row["Printed_Page"],
                    "Column": row["Column"],
                    "Item": row["Item"],
                    "Gloss": row["Gloss"],
                    "Site": site,
                    "Role": "target",
                    "Source_Name": metadata["name"],
                    "Location": metadata["location"],
                    "Source_Raw": raw,
                    "Manual_Transcription": raw,
                    "Resolved_Forms": " | ".join(variant["form"] for variant in forms),
                    "Similarity_Groups": " | ".join(variant["group"] for variant in forms),
                    "Qualifiers": " | ".join(variant["qualifier"] for variant in forms),
                    "Manual_Review": row["Review"],
                    "Item_Review_Note": review_note,
                    "Uncertainty_Type": review_type,
                    "OCR_Evidence_Primary": evidence["Raw_OCR_Primary"],
                    "OCR_Evidence_Secondary": evidence["Raw_OCR_Secondary"],
                    "OCR_Evidence_Latin": evidence["Raw_OCR_Latin"],
                    "Source_Status": status,
                    "Status": install_status,
                    "Reason": reason,
                    "Language_ID": LANGUAGE_ID,
                    "Dialect_ID": dialect_id(site),
                    "Citation": cite,
                    "Entry_Keys": " | ".join(entry_keys),
                })
            else:
                audit.append({
                    "Record_Type": "wordlist cell",
                    "Source_Key": SOURCE_KEY,
                    "PDF_Page": evidence["PDF_Page"],
                    "Printed_Page": evidence["Printed_Page"],
                    "Column": evidence["Column"],
                    "Item": evidence["Item"],
                    "Gloss": row["Gloss"],
                    "Site": site,
                    "Role": "comparison",
                    "Source_Name": metadata["name"],
                    "Location": metadata["location"],
                    "Source_Raw": "",
                    "Manual_Transcription": "",
                    "Resolved_Forms": "",
                    "Similarity_Groups": "",
                    "Qualifiers": "",
                    "Manual_Review": "excluded comparison; not manually transcribed",
                    "Item_Review_Note": "",
                    "Uncertainty_Type": "",
                    "OCR_Evidence_Primary": evidence["Raw_OCR_Primary"],
                    "OCR_Evidence_Secondary": evidence["Raw_OCR_Secondary"],
                    "OCR_Evidence_Latin": evidence["Raw_OCR_Latin"],
                    "Source_Status": "excluded-comparison-untranscribed",
                    "Status": "excluded",
                    "Reason": (
                        "comparison list outside the six Haryanvi target lists; exact cell locator "
                        "and non-authoritative OCR evidence retained for source accounting"
                    ),
                    "Language_ID": "",
                    "Dialect_ID": "",
                    "Citation": cite,
                    "Entry_Keys": "",
                })

    if len(audit) != 2100:
        raise AssertionError(f"source-cell audit topology drift: {len(audit)}")
    if (direct_response_cells, cross_reference_cells, target_blank_cells, target_elicitation_cells) != (
        1231, 7, 21, 1
    ):
        raise AssertionError("target source-status topology drift")
    if direct_variants != 1546 or len(installed) != 1553:
        raise AssertionError(
            f"installed variant topology drift: direct={direct_variants} installed={len(installed)}"
        )
    if Counter(row["Status"] for row in audit) != Counter({"installed": 1238, "excluded": 862}):
        raise AssertionError("audit status topology drift")
    if len({row["Entry_Key"] for row in installed}) != len(installed):
        raise AssertionError("duplicate installed Entry_Key")
    if any(not row["Form"] or row["Form"] != row["Phonemic"] for row in installed):
        raise AssertionError("blank or mismatched installed source transcription")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in installed):
        raise AssertionError("non-NFC installed form")
    if any(row["Parameter_ID"] or row["Cognateset"] or row["Etymology"] for row in installed):
        raise AssertionError("source similarity groups must not become etymological claims")
    if any("tesseract" in row["Notes"].lower() or "ocr" in row["Notes"].lower() for row in installed):
        raise AssertionError("OCR evidence must not be represented as installed transcription")

    manifest = {
        "source_key": SOURCE_KEY,
        "publisher_file": "JLSR2024_011.pdf",
        "publisher_file_sha256": SOURCE_SHA256,
        "publisher_pdf_url": (
            "https://www.sil.org/system/files/reapdata/39/32/25/"
            "39322561837182664281680018892812535354/JLSR2024_011.pdf"
        ),
        "source_extent": "89 PDF pages; Appendix A.3 printed pp. 21-34 (PDF pp. 28-41)",
        "manual_transcription_file": str(MANUAL.relative_to(REPO)),
        "manual_transcription_sha256": MANUAL_SHA256,
        "ocr_scaffold_file": str(OCR_SCAFFOLD.relative_to(REPO)),
        "ocr_scaffold_sha256": OCR_SCAFFOLD_SHA256,
        "source_prompts": 210,
        "source_lists": 10,
        "source_cells": len(audit),
        "target_lists": len(TARGET_SITES),
        "target_cells": 1260,
        "manually_reviewed_target_cells": 1260,
        "direct_target_response_cells": direct_response_cells,
        "direct_target_variants": direct_variants,
        "cross_reference_cells": cross_reference_cells,
        "target_blank_cells": target_blank_cells,
        "target_elicitation_note_cells": target_elicitation_cells,
        "installed_responses": len(installed),
        "comparison_lists": len(COMPARISON_SITES),
        "comparison_cells": 840,
        "comparison_cells_manually_transcribed": 0,
        "comparison_cells_with_primary_ocr_evidence": sum(
            bool(row["OCR_Evidence_Primary"]) for row in audit if row["Role"] == "comparison"
        ),
        "audit_records": len(audit),
        "audit_status_counts": dict(Counter(row["Status"] for row in audit)),
        "manual_item_review_notes": sum(bool(row["Uncertainty"]) for row in manual),
        "target_cells_with_typed_uncertainty": sum(
            bool(row["Uncertainty_Type"]) for row in audit if row["Role"] == "target"
        ),
        "replacement_or_private_use_glyphs": sum(
            "�" in row["Form"] or any(0xE000 <= ord(char) <= 0xF8FF for char in row["Form"])
            for row in installed
        ),
        "unparsed_target_cells": 0,
        "ocr_used": True,
        "ocr_policy": (
            "Tesseract is structural/audit evidence only. All 1,260 target cells and every "
            "installed IPA form were manually read from the scan. The 840 excluded comparison "
            "cells were not manually transcribed and contribute no installed form. "
            "No installed form originates from OCR."
        ),
        "etymology_edges": 0,
        "coordinate_policy": (
            "the report names source localities but gives no point coordinates; dialect "
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
    print(
        f"installed={len(installed)} source_cells={len(audit)} "
        f"excluded={sum(row['Status'] == 'excluded' for row in audit)}"
    )


if __name__ == "__main__":
    write()
