#!/usr/bin/env python3
"""Install the image-only wordlists in SIL ESR 2019-005.

The publisher PDF is not redistributed.  Tesseract provides a reproducible
layout scaffold, while ``transcription_pass2_quarter.tsv`` is the checked
source-facing transcription made from quarter-column crops of the 168-dpi
page images.  This importer treats the seven Mudhili Gadaba lists as targets
and the Telugu list as an excluded comparison control.
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
TRANSCRIPTION = HERE / "transcription_pass2_quarter.tsv"
INSTALLED = FORMS / "20260828-sil-mudhili-gadaba.csv"
AUDIT = RAW_ROOT / "20260828-sil-mudhili-gadaba-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-mudhili-gadaba-manifest.json"

SOURCE_KEY = "adimathara2019mudhili"
SOURCE_PDF = "silesr2019_005.pdf"
SOURCE_SHA256 = "f5fd88b84e1add2509314186bbde779e35e6675c96390c885d712ffee39b9300"
KEY_PREFIX = "silgadaba2019"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Report", "PDF_Page", "Printed_Page", "Column", "Gloss_Number",
    "Gloss", "Site_Code", "Site_Name", "Comparison_Role", "Response", "Group",
    "Transcription", "Review", "Uncertainty", "Status", "Reason", "Language_ID",
    "Dialect_ID", "Source", "Entry_Key",
]

SITES = {
    "Bobbili": ("Bobbilivalasa", "target", "Pachipenta mandal"),
    "Gogada": ("Gogaduvalasa", "target", "Pachipenta mandal"),
    "Panuku": ("Panukuvalasa", "target", "Salur mandal"),
    "Reyavani": ("Reyavanivalasa", "target", "Salur mandal"),
    "Kotha": ("Kothavalasa", "target", "Salur mandal"),
    "Suregadi": ("Suregadivalasa", "target", "Pachipenta mandal"),
    "Chinachipuru": ("Chinachipuruvalasa", "target", "Pachipenta mandal"),
    "TELUGU": ("Srikakulam Telugu", "comparison control", "Srikakulam district"),
}

DISQUALIFIED_ITEMS = {11, 23, 32, 70, 188}
ITEM = re.compile(r"^#(\d+)\s+(.+)$")
MARKER = re.compile(r"^@@ p(\d+)c([123])$")


def slug(value: str) -> str:
    return "-".join("".join(c if c.isalnum() else " " for c in value.lower()).split())


def dialect_id(site: str) -> str:
    return f"sil-gadaba-2019-{slug(SITES[site][0])}"


def dialect_tag(site: str) -> str:
    did = dialect_id(site)
    name = SITES[site][0]
    return f"dialect:Gadaba:{quote(did, safe='')}:{quote(name, safe='')}"


def review_flags(form: str) -> str:
    flags = []
    if any(char in form for char in "ʈɖɳ") or any(mark in form for mark in ("t̪", "d̪", "n̪")):
        flags.append("source-raster-coronal")
    if any(char in form for char in "ʌəɛɪʊɐ"):
        flags.append("source-raster-vowel")
    if "ⁱ" in form:
        flags.append("source-raster-superscript")
    if "̃" in unicodedata.normalize("NFD", form):
        flags.append("source-raster-nasalization")
    if "?" in form:
        flags.append("source-raster-unresolved")
    return ";".join(flags)


def installable_form(form: str) -> tuple[str, str]:
    """Separate source annotations from the segment string used by CLDF."""
    annotations = []
    for label in ("(sg)", "(pl)"):
        if label in form:
            annotations.append(f"source marks {label[1:-1]}")
            form = form.replace(label, "")
    if "?" in form:
        annotations.append("source transcription carries a question mark")
        form = form.replace("?", "")
    return form, "; ".join(annotations)


def parse() -> tuple[list[dict], list[dict]]:
    records = []
    exclusions = []
    item = gloss = site = None
    pdf_page = printed_page = column = None
    responses = Counter()
    item_headers = []
    for line_number, raw in enumerate(TRANSCRIPTION.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        marker = MARKER.fullmatch(line)
        if marker:
            page_index, column = map(int, marker.groups())
            pdf_page = page_index + 1
            printed_page = page_index - 3
            continue
        header = ITEM.fullmatch(line)
        if header:
            item = int(header.group(1))
            gloss = header.group(2).strip()
            item_headers.append(item)
            site = None
            continue
        if line == "DISQUALIFIED":
            exclusions.append({
                "item": item, "gloss": gloss, "pdf_page": pdf_page,
                "printed_page": printed_page, "column": column,
            })
            continue
        fields = raw.split("\t")
        if len(fields) != 3:
            raise AssertionError(f"malformed transcription line {line_number}: {raw!r}")
        site_field, group, form = fields
        if site_field != "-":
            if site_field not in SITES:
                raise AssertionError(f"unknown site at line {line_number}: {site_field}")
            site = site_field
        if site is None or item is None or pdf_page is None:
            raise AssertionError(f"orphan transcription line {line_number}: {raw!r}")
        if unicodedata.normalize("NFC", form) != form:
            raise AssertionError(f"non-NFC transcription at line {line_number}: {form!r}")
        responses[item, site] += 1
        records.append({
            "line": line_number, "item": item, "gloss": gloss, "site": site,
            "group": group, "form": form, "response": responses[item, site],
            "pdf_page": pdf_page, "printed_page": printed_page, "column": column,
        })

    if item_headers != list(range(1, 211)):
        raise AssertionError(f"item topology drift: {item_headers}")
    if {row["item"] for row in exclusions} != DISQUALIFIED_ITEMS:
        raise AssertionError(f"disqualified-item drift: {exclusions}")
    for item_number in set(range(1, 211)) - DISQUALIFIED_ITEMS:
        missing_sites = set(SITES) - {row["site"] for row in records if row["item"] == item_number}
        if missing_sites:
            raise AssertionError(f"item {item_number} lacks source response rows for {sorted(missing_sites)}")
    return records, exclusions


def locator(record: dict, site_name: str) -> str:
    return (
        f"{SOURCE_KEY}[Appendix A.3, printed p. {record['printed_page']}, "
        f"item {record['item']}, {site_name}]"
    )


def build() -> tuple[list[dict], list[dict], dict]:
    records, exclusions = parse()
    installed = []
    audit = []
    for record in records:
        site_name, role, mandal = SITES[record["site"]]
        source = locator(record, site_name)
        target = role == "target"
        missing = record["form"] == "No Entry"
        status = "installed" if target and not missing else "excluded"
        reason = ""
        if not target:
            reason = "Telugu comparison control; not a target Mudhili Gadaba list"
        elif missing:
            reason = "source explicitly prints No Entry"
        did = dialect_id(record["site"]) if target else ""
        key = ""
        if status == "installed":
            key = (
                f"{KEY_PREFIX}:g{record['item']:03d}:{did}:"
                f"i{record['response']}"
            )
            form, source_annotation = installable_form(record["form"])
            notes = f"Appendix A.3 lexical-similarity group {record['group']}; {mandal}"
            if source_annotation:
                notes += f"; {source_annotation}"
            flags = review_flags(record["form"])
            if flags:
                notes += f"; source-raster review flags: {flags}"
            installed.append({
                "Language_ID": "Gadaba", "Parameter_ID": "", "Form": form,
                "Gloss": record["gloss"], "Native": "", "Phonemic": record["form"],
                "Notes": notes, "Source": source, "Cognateset": "", "Etymology": "",
                "Entry_Key": key, "Variant_Of_Key": "", "Borrowed_From_Key": "",
                "Derivation_Parent_Keys": "", "Tags": dialect_tag(record["site"]),
            })
        audit.append({
            "Record_Type": "wordlist response", "Report": SOURCE_KEY,
            "PDF_Page": record["pdf_page"], "Printed_Page": record["printed_page"],
            "Column": record["column"], "Gloss_Number": record["item"],
            "Gloss": record["gloss"], "Site_Code": record["site"], "Site_Name": site_name,
            "Comparison_Role": role, "Response": record["response"], "Group": record["group"],
            "Transcription": record["form"], "Review": "visually transcribed at quarter-column zoom",
            "Uncertainty": review_flags(record["form"]), "Status": status, "Reason": reason,
            "Language_ID": "Gadaba" if target else "", "Dialect_ID": did,
            "Source": source, "Entry_Key": key,
        })

    for record in exclusions:
        audit.append({
            "Record_Type": "item exclusion", "Report": SOURCE_KEY,
            "PDF_Page": record["pdf_page"], "Printed_Page": record["printed_page"],
            "Column": record["column"], "Gloss_Number": record["item"], "Gloss": record["gloss"],
            "Site_Code": "", "Site_Name": "", "Comparison_Role": "all lists",
            "Response": "", "Group": "", "Transcription": "", "Review": "classified",
            "Uncertainty": "", "Status": "excluded", "Reason": "source prints DISQUALIFIED",
            "Language_ID": "", "Dialect_ID": "", "Source": (
                f"{SOURCE_KEY}[Appendix A.3, printed p. {record['printed_page']}, item {record['item']}]"
            ), "Entry_Key": "",
        })

    if len(records) != 1760 or len(installed) != 1538 or len(audit) != 1765:
        raise AssertionError(
            f"topology drift: records={len(records)} installed={len(installed)} audit={len(audit)}"
        )
    status_counts = Counter(row["Status"] for row in audit)
    manifest = {
        "source_key": SOURCE_KEY, "publisher_file": SOURCE_PDF,
        "publisher_file_sha256": SOURCE_SHA256,
        "transcription_file": str(TRANSCRIPTION.relative_to(REPO)),
        "transcription_sha256": hashlib.sha256(TRANSCRIPTION.read_bytes()).hexdigest(),
        "items": 210, "disqualified_items": sorted(DISQUALIFIED_ITEMS),
        "source_response_records": len(records), "target_installed": len(installed),
        "target_no_entry": sum(
            row["Status"] == "excluded" and row["Reason"] == "source explicitly prints No Entry"
            for row in audit
        ),
        "control_records": sum(row["Comparison_Role"] == "comparison control" for row in audit),
        "audit_records": len(audit), "audit_status_counts": dict(status_counts),
        "uncertainty_counts": dict(Counter(
            flag for row in audit for flag in row["Uncertainty"].split(";") if flag
        )),
        "etymology_edges": 0,
    }
    return installed, audit, manifest


def write() -> None:
    installed, audit, manifest = build()
    with INSTALLED.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FORM_FIELDS)
        # Manual form inputs use the repository's headerless rich 15-column
        # schema.  The audit is intentionally headered, but this file is not.
        writer.writerows(installed)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader(); writer.writerows(audit)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"installed={len(installed)} target_gaps={manifest['target_no_entry']} "
        f"controls={manifest['control_records']} audit={len(audit)}"
    )


if __name__ == "__main__":
    write()
