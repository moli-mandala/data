#!/usr/bin/env python3
"""Build the manually reviewed word lists in JLSR 2021-029.

``manual_review_data.py`` is one row per photographed page/lect column.  Each
array has exactly the source cells visible in that column.  The importer expands
the arrays into a complete per-cell audit and splits source-marked slash
alternatives into stable child rows.  OCR is retained only as comparison evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent


def _load_manual_rows():
    """Load the sibling ledger under a source-unique module name."""
    path = HERE / "manual_review_data.py"
    spec = importlib.util.spec_from_file_location("sil_koya_2021_manual_review_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load manual review ledger: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ROWS


ROWS = _load_manual_rows()
REPO = HERE.parents[4]
FORMS = REPO / "data/other/forms"
PROMPTS = HERE / "prompts.tsv"
OCR = HERE / "tesseract_raw.txt"
INSTALLED = FORMS / "20260828-sil-koya.csv"
AUDIT = FORMS / "raw_data/20260828-sil-koya-audit.csv"
MANIFEST = FORMS / "raw_data/20260828-sil-koya-manifest.json"

SOURCE_KEY = "devagnanavaram-et-al2021koya"
SOURCE_SHA256 = "a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d"
KEY_PREFIX = "silkoya1985"

# Code, display name, role, dialect-group, PDF start page, source item ranges.
SITES = {
    "JAG": ("Jaganathapuram Koya", "target", "eastern Koya"),
    "CHI": ("Chintoor Koya", "target", "eastern Koya"),
    "POD": ("Podia Koya", "target", "eastern Koya"),
    "UTN": ("Utnoor Gondi", "target", "western Gondi"),
    "BHG": ("Bhamani Gondi", "target", "western Gondi"),
    "BHM": ("Bhamani Madia", "target", "western Madia"),
    "MAL": ("Malakanagiri Koya", "target", "eastern Koya"),
    "TEL": ("Telugu", "comparison control", ""),
    "ORI": ("Oriya", "comparison control", ""),
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Item", "Gloss", "Site_Code", "Site_Name", "Role", "Dialect_Group",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription", "Manual_Review",
    "Uncertainty", "OCR_Evidence", "Alternate_Digital_Evidence", "Status", "Reason",
    "Language_ID", "Dialect_ID", "Source", "Entry_Keys",
]


def slug(value: str) -> str:
    return "-".join("".join(c if c.isalnum() else " " for c in value.lower()).split())


def dialect_id(code: str) -> str:
    if code in {"BHG", "BHM"}:
        return f"sil-koya-1985-{slug(SITES[code][0])}"
    return f"sil-koya-1985-{slug(SITES[code][0].removesuffix(' Koya').removesuffix(' Gondi').removesuffix(' Madia'))}"


def dialect_tag(code: str) -> str:
    name = SITES[code][0]
    did = dialect_id(code)
    return f"dialect:Gondi:{quote(did, safe='')}:{quote(name, safe='')}"


def read_prompts() -> dict[int, str]:
    with PROMPTS.open(encoding="utf-8", newline="") as stream:
        rows = {int(r["Item"]): r["Gloss"] for r in csv.DictReader(stream, delimiter="\t")}
    if set(rows) != set(range(1, 211)):
        raise AssertionError("prompts.tsv must contain exactly items 1-210")
    return rows


def read_pages() -> dict[tuple[int, str], dict]:
    pages = {}
    for source_row in ROWS:
        row = dict(source_row)
        page = int(row["PDF_Page"])
        code = row["Site"]
        if code not in SITES:
            raise AssertionError(f"unknown site {code}")
        key = (page, code)
        if key in pages:
            raise AssertionError(f"duplicate manual page/column {key}")
        if row["Review"] != "manual-source-image":
            raise AssertionError(f"unfinished source-image review at {key}")
        forms = [unicodedata.normalize("NFC", form) for form in row["Forms"]]
        first = int(row["First_Item"])
        expected = 10 if first == 201 else 20
        if len(forms) != expected:
            raise AssertionError(f"{key} has {len(forms)} cells, expected {expected}")
        row["forms"] = forms
        row["uncertainties"] = list(row.get("Uncertainties", []))
        if not row["uncertainties"]:
            row["uncertainties"] = [""] * expected
        for index, form in enumerate(forms):
            if not form and not row["uncertainties"][index]:
                row["uncertainties"][index] = "confirmed ruled blank"
        row["digital"] = row.get("Alternate_Digital", [])
        for field in ("uncertainties", "digital"):
            if row[field] and len(row[field]) != expected:
                raise AssertionError(f"{key} {field} length does not match cells")
        pages[key] = row
    return pages


def expected_page_item_pairs() -> dict[tuple[int, str], tuple[int, int]]:
    expected = {}
    for page, first in zip(range(82, 92), range(1, 211, 20)):
        for code in ("JAG", "CHI", "POD"):
            expected[(page, code)] = (first, 1 if code == "JAG" else 2 if code == "CHI" else 3)
    for page, first in zip(range(92, 103), range(1, 211, 20)):
        for code in ("UTN", "BHG", "BHM"):
            expected[(page, code)] = (first, 1 if code == "UTN" else 2 if code == "BHG" else 3)
    for page, first in zip(range(103, 113), [1, 21, 41, 81, 101, 121, 141, 161, 181, 201]):
        expected[(page, "MAL")] = (first, 1)
    for page, first in zip(range(113, 124), range(1, 211, 20)):
        expected[(page, "TEL")] = (first, 1)
        expected[(page, "ORI")] = (first, 2)
    return expected


def split_variants(value: str) -> list[str]:
    """Split source slash alternatives, but never slashes within parentheses."""
    values, start, depth = [], 0, 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        elif char == "/" and depth == 0:
            values.append(value[start:index].strip())
            start = index + 1
    values.append(value[start:].strip())
    return [value for value in values if value]


def locator(page: int, item: int, site_name: str) -> str:
    printed = page - 5
    return (
        f"{SOURCE_KEY}[Appendix E, printed p. {printed}, item {item}, {site_name}]"
    )


def build() -> tuple[list[list[str]], list[dict], dict]:
    prompts = read_prompts()
    pages = read_pages()
    expected = expected_page_item_pairs()
    if set(pages) != set(expected):
        raise AssertionError(
            f"manual page topology drift: missing={sorted(set(expected)-set(pages))[:5]}, "
            f"extra={sorted(set(pages)-set(expected))[:5]}"
        )
    forms: list[list[str]] = []
    audit: list[dict] = []
    seen_cells = set()
    for (page, code), (first, column) in sorted(expected.items()):
        row = pages[(page, code)]
        if int(row["First_Item"]) != first or int(row["Column"]) != column:
            raise AssertionError(f"page metadata mismatch at {(page, code)}")
        site_name, role, group = SITES[code]
        for offset, transcription in enumerate(row["forms"]):
            item = first + offset
            cell = (item, code)
            if cell in seen_cells:
                raise AssertionError(f"duplicate source cell {cell}")
            seen_cells.add(cell)
            uncertainty = row["uncertainties"][offset] if row["uncertainties"] else ""
            digital = row["digital"][offset] if row["digital"] else ""
            source = locator(page, item, site_name)
            entry = {
                "Record_Type": "wordlist cell", "Item": item, "Gloss": prompts[item],
                "Site_Code": code, "Site_Name": site_name, "Role": role,
                "Dialect_Group": group, "PDF_Page": page, "Printed_Page": page - 5,
                "Column": column, "Manual_Transcription": transcription,
                "Manual_Review": row["Review"], "Uncertainty": uncertainty,
                "OCR_Evidence": f"tesseract_raw.txt#pdf{page}",
                "Alternate_Digital_Evidence": digital, "Status": "", "Reason": "",
                "Language_ID": "", "Dialect_ID": "", "Source": source, "Entry_Keys": "",
            }
            if role != "target":
                entry.update(Status="excluded", Reason=f"excluded {role}")
                audit.append(entry)
                continue
            entry.update(Language_ID="Gondi", Dialect_ID=dialect_id(code))
            if not transcription:
                entry.update(Status="missing", Reason="source prints a blank or ruled missing cell")
                audit.append(entry)
                continue
            variants = split_variants(transcription)
            keys = []
            for variant_index, variant in enumerate(variants, 1):
                key = f"{KEY_PREFIX}:{code.lower()}:i{item:03d}:v{variant_index}"
                keys.append(key)
                notes = f"Appendix E; {group}; manual source-image transcription"
                tags = dialect_tag(code)
                if uncertainty:
                    notes += f"; review flag: {uncertainty}"
                    tags += " uncertain"
                forms.append([
                    "Gondi", "", variant, prompts[item], "", variant, notes, source, "", "",
                    key, "", "", "", tags,
                ])
            entry.update(Status="installed", Entry_Keys="|".join(keys))
            audit.append(entry)
    # MAL items 61-80 are absent from the report and explicitly noted by the editor.
    for item in range(61, 81):
        code = "MAL"
        site_name, role, group = SITES[code]
        source = f"{SOURCE_KEY}[Appendix E, editor note after printed p. 100, item {item}, {site_name}]"
        audit.append({
            "Record_Type": "wordlist cell", "Item": item, "Gloss": prompts[item],
            "Site_Code": code, "Site_Name": site_name, "Role": role, "Dialect_Group": group,
            "PDF_Page": 105, "Printed_Page": 100, "Column": 1,
            "Manual_Transcription": "", "Manual_Review": "manual-source-image",
            "Uncertainty": "", "OCR_Evidence": "tesseract_raw.txt#pdf105",
            "Alternate_Digital_Evidence": "", "Status": "missing",
            "Reason": "items 61-80 are absent; explicit editor note in Appendix E",
            "Language_ID": "Gondi", "Dialect_ID": dialect_id(code), "Source": source,
            "Entry_Keys": "",
        })
    # The three eastern lists stop at item 200; no pronoun page was printed.
    for code in ("JAG", "CHI", "POD"):
        site_name, role, group = SITES[code]
        for item in range(201, 211):
            source = f"{SOURCE_KEY}[Appendix E, list ends at printed p. 86, omitted item {item}, {site_name}]"
            audit.append({
                "Record_Type": "omitted list slot", "Item": item, "Gloss": prompts[item],
                "Site_Code": code, "Site_Name": site_name, "Role": role,
                "Dialect_Group": group, "PDF_Page": 91, "Printed_Page": 86,
                "Column": 1 if code == "JAG" else 2 if code == "CHI" else 3,
                "Manual_Transcription": "", "Manual_Review": "manual-source-image",
                "Uncertainty": "", "OCR_Evidence": "tesseract_raw.txt#pdf91",
                "Alternate_Digital_Evidence": "", "Status": "missing",
                "Reason": "list ends at item 200; no items 201-210 are printed",
                "Language_ID": "Gondi", "Dialect_ID": dialect_id(code),
                "Source": source, "Entry_Keys": "",
            })
    if len(seen_cells) + 20 + 30 != 9 * 210:
        raise AssertionError(
            f"accounted for {len(seen_cells)+20+30} list slots, expected 1890"
        )
    audit.sort(key=lambda r: (int(r["Item"]), r["Site_Code"]))
    status_counts = Counter(r["Status"] for r in audit)
    uncertainty_counts = Counter(
        r["Uncertainty"] for r in audit if r["Uncertainty"]
    )
    metadata = {
        "source_key": SOURCE_KEY,
        "source_pdf_sha256": SOURCE_SHA256,
        "ocr_scaffold_sha256": hashlib.sha256(OCR.read_bytes()).hexdigest(),
        "manual_review_ledger_sha256": hashlib.sha256(
            (HERE / "manual_review_data.py").read_bytes()
        ).hexdigest(),
        "source_archive_entry": 88873,
        "source_pdf_pages": 124,
        "wordlist_pdf_pages": "80-123",
        "wordlist_printed_pages": "75-118",
        "counts": {
            "conceptual_list_slots": len(audit),
            "source_image_cells_manually_reviewed": len(seen_cells),
            "omitted_slots_accounted_for": 50,
            "installed_rows": len(forms),
            "status": dict(status_counts), "uncertainty": dict(uncertainty_counts),
        },
        "policy": {
            "transcription": "manual review of every cell against embedded source image",
            "ocr": "comparison scaffold only; never installed without manual source-image review",
            "controls": ["TEL", "ORI"],
            "etymology": "none claimed; all forms unlinked",
        },
    }
    return forms, audit, metadata


def write(forms: list[list[str]], audit: list[dict], metadata: dict, install: bool) -> None:
    target = INSTALLED if install else REPO / "tmp/20260828-sil-koya-preview.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)
    MANIFEST.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"source_cells={len(audit)} installed={len(forms)} "
        f"missing={sum(r['Status']=='missing' for r in audit)} "
        f"controls={sum(r['Status']=='excluded' for r in audit)} output={target}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    write(*build(), install=args.install)


if __name__ == "__main__":
    main()
