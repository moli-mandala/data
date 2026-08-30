#!/usr/bin/env python3
"""Build the manually reviewed Kullu wordlists in JLSR 2021-009.

``manual_pages.tsv`` is authoritative. Vision/OCR files are comparison evidence
only and never populate an installed form.
"""

from __future__ import annotations

import argparse
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
MANUAL = HERE / "manual_pages.tsv"
PROMPTS = HERE / "prompts.tsv"
OCR = HERE / "transcription.tsv"
INSTALLED = FORMS / "20260828-sil-kullu.csv"
AUDIT = FORMS / "raw_data/20260828-sil-kullu-audit.csv"
MANIFEST = FORMS / "raw_data/20260828-sil-kullu-manifest.json"

SOURCE_KEY = "blair2021kullu"
SOURCE_SHA256 = "720a97198254160bfa88a9557b33955b2814878e346901ff399cacc53d5c4fdd"
KEY_PREFIX = "silkullu1985"
LANGUAGE_ID = "kul"

# Source column order, name, source label, locality, field date.
SITES = {
    "CHU": ("Churla", "Kullui", "Churla/Lag Valley, Kullu Tehsil", "10 April 1985"),
    "LOR": ("Loren", "Kullui", "Loren/S. Kullu Valley, Kullu Tehsil", "10-11 April 1985"),
    "SHA": ("Shalwar", "Inner Seraji", "Shalwar Village, Banjar Tehsil", "19 April 1985"),
    "CHI": ("Chinninal", "Inner Seraji", "Chinninal Village, Banjar Tehsil", "18 April 1985"),
    "SHG": ("Shangarh", "Inner Seraji", "Shangarh, Banjar Tehsil", "20 April 1985"),
    "MAN": ("Manali", "Kullui", "Manali, Kullu Tehsil", "10 April 1985"),
    "RAI": ("Raila", "Inner Seraji", "Raila Village, Kullu Tehsil", "25 April 1985"),
    "MAR": ("Maraur", "Inner Seraji", "Maraur Village, Banjar Tehsil (?)", "3 May 1985"),
    "SID": ("Sidua", "Inner Seraji", "Sidua, Banjar Tehsil", "3 May 1985"),
    "JIB": ("Jibhi", "Inner Seraji", "Jibhi, Banjar Tehsil", "9 May 1985"),
    "BAT": ("Bathad", "Inner Seraji", "Bathad, Banjar Tehsil", "10 May 1985"),
    "GAR": ("Garsah", "Kullui", "Garsah, Kullu Tehsil", "22 May 1985"),
    "KUL": ("Kullu", "Kullui", "Kullu HQ, Kullu Tehsil", "6 June 1985"),
    "BHU": ("Bhutti", "Kullui", "Bhutti Village (Lag Valley), Kullu Tehsil", "1 June 1985"),
    "MNK": ("Manikaran", "Kullui", "Manikaran, Kullu Tehsil", "21 May 1985"),
    "ANI": ("Ani", "Outer Seraji", "Ani, Ani Tehsil", "17 June 1985"),
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Item", "Gloss", "Site_Code", "Site_Name", "Source_Dialect_Label",
    "PDF_Page", "Printed_Page", "Column", "Cell_Image", "Manual_Transcription",
    "Manual_Review", "Uncertainty", "OCR_Evidence", "Status", "Reason", "Language_ID",
    "Dialect_ID", "Source", "Entry_Keys",
]


def dialect_id(code: str) -> str:
    return f"sil-kullu-1985-{SITES[code][0].lower()}"


def dialect_tag(code: str) -> str:
    name = SITES[code][0]
    did = dialect_id(code)
    return f"dialect:{LANGUAGE_ID}:{quote(did, safe='')}:{quote(name, safe='')}"


def read_prompts() -> dict[int, tuple[str, str]]:
    with PROMPTS.open(encoding="utf-8", newline="") as stream:
        rows = {int(r["Item"]): (r["Gloss"], r["Source_prompt_note"]) for r in csv.DictReader(stream, delimiter="\t")}
    if set(rows) != set(range(1, 199)):
        raise AssertionError("prompts.tsv must contain exactly items 1-198")
    return rows


def read_ocr() -> dict[tuple[int, str], str]:
    evidence = {}
    with OCR.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            pieces = [row["Raw_OCR"].strip(), row["OCR_Alternates"].strip()]
            evidence[(int(row["Item"]), row["Site"])] = " | ".join(p for p in pieces if p)
    if len(evidence) != 3168:
        raise AssertionError(f"OCR scaffold topology drift: {len(evidence)} cells")
    return evidence


def read_manual() -> dict[tuple[int, str], dict]:
    pages = {}
    with MANUAL.open(encoding="utf-8", newline="") as stream:
        for source_row in csv.DictReader(stream, delimiter="\t"):
            row = dict(source_row)
            page, code, first = int(row["PDF_Page"]), row["Site"], int(row["First_Item"])
            if code not in SITES or (first, code) in pages:
                raise AssertionError(f"bad or duplicate manual block {(first, code)}")
            if row["Review"] != "manual-source-image":
                raise AssertionError(f"unfinished source-image review at {(page, code)}")
            forms = json.loads(row["Forms_JSON"])
            uncertainties = json.loads(row["Uncertainty_JSON"])
            expected = 6 if first == 193 else 16
            if len(forms) != expected or len(uncertainties) not in (0, expected):
                raise AssertionError(f"manual array length mismatch at {(page, code)}")
            if not uncertainties:
                uncertainties = [""] * expected
            forms = [unicodedata.normalize("NFC", value) for value in forms]
            for index, form in enumerate(forms):
                if not form and uncertainties[index] != "blank":
                    raise AssertionError(f"blank is not explicitly accounted for at {(code, first + index)}")
            row.update(forms=forms, uncertainties=uncertainties, first=first, page=page)
            pages[(first, code)] = row
    expected = {(first, code) for first in range(1, 194, 16) for code in SITES}
    if set(pages) != expected:
        raise AssertionError(f"manual topology drift: missing={set(expected)-set(pages)} extra={set(pages)-set(expected)}")
    return pages


def split_variants(value: str) -> list[str]:
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


def locator(page: int, item: int, code: str) -> str:
    return f"{SOURCE_KEY}[Appendix C, printed p. {page - 7}, item {item}, {SITES[code][0]}]"


def build() -> tuple[list[list[str]], list[dict], dict]:
    prompts, pages, ocr = read_prompts(), read_manual(), read_ocr()
    forms, audit, seen = [], [], set()
    columns = {code: (index % 3) + 1 for index, code in enumerate(SITES)}
    for first in range(1, 194, 16):
        for code in SITES:
            row = pages[(first, code)]
            for offset, transcription in enumerate(row["forms"]):
                item = first + offset
                cell = (item, code)
                if cell in seen:
                    raise AssertionError(f"duplicate source cell {cell}")
                seen.add(cell)
                gloss, prompt_note = prompts[item]
                source = locator(row["page"], item, code)
                uncertainty = row["uncertainties"][offset]
                entry = {
                    "Record_Type": "wordlist cell", "Item": item, "Gloss": gloss,
                    "Site_Code": code, "Site_Name": SITES[code][0],
                    "Source_Dialect_Label": SITES[code][1], "PDF_Page": row["page"],
                    "Printed_Page": int(row["Printed_Page"]), "Column": columns[code],
                    "Cell_Image": f"cells/i{item:03d}-{code}-pdf{row['page']:03d}.png",
                    "Manual_Transcription": transcription, "Manual_Review": row["Review"],
                    "Uncertainty": uncertainty, "OCR_Evidence": ocr[cell], "Status": "",
                    "Reason": "", "Language_ID": LANGUAGE_ID, "Dialect_ID": dialect_id(code),
                    "Source": source, "Entry_Keys": "",
                }
                if not transcription:
                    entry.update(Status="missing", Reason="source cell visually confirmed blank")
                    audit.append(entry)
                    continue
                variants, keys = split_variants(transcription), []
                first_key = f"{KEY_PREFIX}:{code.lower()}:i{item:03d}:v1"
                for index, variant in enumerate(variants, 1):
                    key = f"{KEY_PREFIX}:{code.lower()}:i{item:03d}:v{index}"
                    keys.append(key)
                    note_bits = [
                        "Appendix C", SITES[code][1], SITES[code][2], SITES[code][3],
                        "manual source-image transcription; OCR comparison only",
                    ]
                    if prompt_note:
                        note_bits.append(f"prompt note: {prompt_note}")
                    if uncertainty:
                        note_bits.append(f"review flag: {uncertainty}")
                    if index > 1:
                        note_bits.append("source-marked slash alternative")
                    tags = dialect_tag(code) + (" uncertain" if uncertainty else "")
                    forms.append([
                        LANGUAGE_ID, "", variant, gloss, "", variant, "; ".join(note_bits),
                        source, "", "", key, first_key if index > 1 else "", "", "", tags,
                    ])
                entry.update(Status="installed", Entry_Keys="|".join(keys))
                audit.append(entry)
    if seen != {(item, code) for item in range(1, 199) for code in SITES}:
        raise AssertionError(f"cell topology drift: {len(seen)}")
    audit.append({
        "Record_Type": "layout-header", "Item": "", "Gloss": "", "Site_Code": "",
        "Site_Name": "", "Source_Dialect_Label": "Hindi", "PDF_Page": 34,
        "Printed_Page": 27, "Column": 1, "Cell_Image": "", "Manual_Transcription": "",
        "Manual_Review": "manual-source-image", "Uncertainty": "", "OCR_Evidence": "",
        "Status": "excluded", "Reason": "Hindi labels the otherwise blank item-number gutter; no Hindi lexical response column exists",
        "Language_ID": "", "Dialect_ID": "", "Source": f"{SOURCE_KEY}[Appendix C, printed p. 27, layout header]",
        "Entry_Keys": "",
    })
    status = Counter(row["Status"] for row in audit)
    manifest = {
        "source_key": SOURCE_KEY,
        "source_url": "https://www.sil.org/resources/archives/88003",
        "source_pdf_url": "https://www.sil.org/system/files/reapdata/40/36/65/40366594353299829635166637372657641345/JLSR2021_009.pdf",
        "source_pdf_sha256": SOURCE_SHA256, "source_pdf_pages": 126,
        "counts": {
            "prompts": 198, "sites": 16, "source_cells": 3168,
            "source_image_cells_manually_reviewed": 3168,
            "installed_cells": status["installed"], "missing_blank_cells": status["missing"],
            "layout_records": status["excluded"], "installed_rows_after_slash_expansion": len(forms),
        },
        "policy": {
            "manual_review": "every source response cell inspected against the source image",
            "ocr": "comparison scaffold only; no form is installed from OCR without manual image verification",
            "similarity_numbers": "excluded from forms; report-specific lexical-similarity group labels are not historical cognacy claims",
            "hindi_header": "layout label in blank item-number gutter, not a response column",
        },
        "uncertainty_counts": dict(sorted(Counter(r["Uncertainty"] for r in audit if r["Uncertainty"]).items())),
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (MANUAL, PROMPTS, OCR)
        },
    }
    return forms, audit, manifest


def write(forms: list[list[str]], audit: list[dict], manifest: dict) -> None:
    INSTALLED.parent.mkdir(parents=True, exist_ok=True)
    with INSTALLED.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader(); writer.writerows(audit)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true", help="write installed CSV, audit, and manifest")
    args = parser.parse_args()
    forms, audit, manifest = build()
    if args.install:
        write(forms, audit, manifest)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
