#!/usr/bin/env python3
"""Build JLSR 2021-040 Appendix F from its manually reviewed cell ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent


def _load_manual_rows():
    """Load the sibling ledger under a source-unique module name."""
    path = HERE / "manual_review_data.py"
    spec = importlib.util.spec_from_file_location("sil_korku_2021_manual_review_data", path)
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
INSTALLED = FORMS / "20260828-sil-korku.csv"
AUDIT = FORMS / "raw_data/20260828-sil-korku-audit.csv"
MANIFEST = FORMS / "raw_data/20260828-sil-korku-manifest.json"

SOURCE_KEY = "stahl2021korku"
SOURCE_SHA256 = "d17426da3788d66c95f05824483941e7d5468e154c66d43c6354262fda00190d"
KEY_PREFIX = "silkorku1985"

# code: (source display name, role)
SITES = {
    "CHI": ("Chikli Ruma", "target"),
    "KHA": ("Khanapur Ruma", "target"),
    "BAG": ("Bagdara Ruma", "target"),
    "WAR": ("Warsari Ruma", "target"),
    "MOR": ("Moragao Bouriya", "target"),
    "LAH": ("Lahi Bouriya", "target"),
    "AMD": ("Amdhana Mawasi", "target"),
    "KHM": ("Khamalpur Bondoy", "target"),
    "NIH": ("Jammat Jalgaon Nihali", "comparison control"),
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Item", "Gloss", "Site_Code", "Site_Name", "Role",
    "PDF_Page", "Printed_Page", "Column", "Manual_Transcription", "Manual_Review",
    "Uncertainty", "OCR_Evidence", "Status", "Reason", "Language_ID",
    "Dialect_ID", "Source", "Entry_Keys",
]


def slug(value: str) -> str:
    return "-".join("".join(c if c.isalnum() else " " for c in value.lower()).split())


def dialect_id(code: str) -> str:
    return f"sil-korku-1985-{slug(SITES[code][0])}"


def dialect_tag(code: str) -> str:
    name = SITES[code][0]
    did = dialect_id(code)
    return f"dialect:ko:{quote(did, safe='')}:{quote(name, safe='')}"


def read_prompts() -> dict[int, str]:
    with PROMPTS.open(encoding="utf-8", newline="") as stream:
        rows = {int(row["Item"]): row["Gloss"] for row in csv.DictReader(stream, delimiter="\t")}
    if set(rows) != set(range(1, 211)):
        raise AssertionError("prompts.tsv must contain exactly items 1-210")
    return rows


def read_pages() -> dict[tuple[int, str], dict]:
    pages = {}
    for source_row in ROWS:
        row = dict(source_row)
        key = (int(row["PDF_Page"]), row["Site"])
        if row["Site"] not in SITES or key in pages:
            raise AssertionError(f"unknown or duplicate manual page {key}")
        if row["Review"] != "manual-source-image":
            raise AssertionError(f"unfinished manual review at {key}")
        expected = 10 if int(row["First_Item"]) == 201 else 20
        if len(row["Forms"]) != expected or len(row["Uncertainties"]) != expected:
            raise AssertionError(f"cell count drift at {key}")
        row["forms"] = [unicodedata.normalize("NFC", value) for value in row["Forms"]]
        row["uncertainties"] = list(row["Uncertainties"])
        for index, form in enumerate(row["forms"]):
            if not form and not row["uncertainties"][index]:
                raise AssertionError(f"unaccounted blank at {key}, offset {index}")
        pages[key] = row
    return pages


def expected_page_item_pairs() -> dict[tuple[int, str], tuple[int, int]]:
    expected = {}
    for start, pair in ((46, ("CHI", "KHA")), (57, ("BAG", "WAR")),
                        (68, ("MOR", "LAH")), (79, ("AMD", "KHM"))):
        for offset, first in enumerate(range(1, 211, 20)):
            for column, code in enumerate(pair, 1):
                expected[(start + offset, code)] = (first, column)
    for offset, first in enumerate(range(1, 211, 20)):
        expected[(90 + offset, "NIH")] = (first, 1)
    return expected


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


def locator(page: int, item: int, site_name: str) -> str:
    return f"{SOURCE_KEY}[Appendix F, printed p. {page - 5}, item {item}, {site_name}]"


def build() -> tuple[list[list[str]], list[dict], dict]:
    prompts = read_prompts()
    pages = read_pages()
    expected = expected_page_item_pairs()
    if set(pages) != set(expected):
        raise AssertionError(
            f"manual page topology drift: missing={sorted(set(expected)-set(pages))[:5]}, "
            f"extra={sorted(set(pages)-set(expected))[:5]}"
        )
    forms, audit, seen_cells = [], [], set()
    for (page, code), (first, column) in sorted(expected.items()):
        row = pages[(page, code)]
        if int(row["First_Item"]) != first or int(row["Column"]) != column:
            raise AssertionError(f"page metadata mismatch at {(page, code)}")
        site_name, role = SITES[code]
        for offset, transcription in enumerate(row["forms"]):
            item = first + offset
            if (code, item) in seen_cells:
                raise AssertionError(f"duplicate source cell {(code, item)}")
            seen_cells.add((code, item))
            uncertainty = row["uncertainties"][offset]
            source = locator(page, item, site_name)
            entry = {
                "Record_Type": "wordlist cell", "Item": item, "Gloss": prompts[item],
                "Site_Code": code, "Site_Name": site_name, "Role": role,
                "PDF_Page": page, "Printed_Page": page - 5, "Column": column,
                "Manual_Transcription": transcription, "Manual_Review": row["Review"],
                "Uncertainty": uncertainty, "OCR_Evidence": f"tesseract_raw.txt#pdf{page}",
                "Status": "", "Reason": "", "Language_ID": "", "Dialect_ID": "",
                "Source": source, "Entry_Keys": "",
            }
            if role != "target":
                entry.update(Status="excluded", Reason="excluded comparison control", Language_ID="Ni")
                audit.append(entry)
                continue
            entry.update(Language_ID="ko", Dialect_ID=dialect_id(code))
            if not transcription:
                reason = ("source response is illegible/clipped and unresolved" if "illegible" in uncertainty
                          else "source prints a ruled blank")
                entry.update(Status="missing", Reason=reason)
                audit.append(entry)
                continue
            keys = []
            for variant_index, variant in enumerate(split_variants(transcription), 1):
                key = f"{KEY_PREFIX}:{code.lower()}:i{item:03d}:v{variant_index}"
                keys.append(key)
                notes = "Appendix F; manual source-image transcription"
                tags = dialect_tag(code)
                if uncertainty:
                    notes += f"; review flag: {uncertainty}"
                    tags += " uncertain"
                forms.append([
                    "ko", "", variant, prompts[item], "", variant, notes, source, "", "",
                    key, "", "", "", tags,
                ])
            entry.update(Status="installed", Entry_Keys="|".join(keys))
            audit.append(entry)
    if len(seen_cells) != 9 * 210:
        raise AssertionError(f"accounted for {len(seen_cells)} cells, expected 1890")
    audit.sort(key=lambda row: (int(row["Item"]), row["Site_Code"]))
    status_counts = Counter(row["Status"] for row in audit)
    uncertainty_counts = Counter(row["Uncertainty"] for row in audit if row["Uncertainty"])
    metadata = {
        "source_key": SOURCE_KEY,
        "source_pdf_sha256": SOURCE_SHA256,
        "ocr_scaffold_sha256": hashlib.sha256(OCR.read_bytes()).hexdigest(),
        "manual_review_ledger_sha256": hashlib.sha256((HERE / "manual_review_data.py").read_bytes()).hexdigest(),
        "source_archive_entry": 90546,
        "source_pdf_pages": 102,
        "wordlist_pdf_pages": "46-100",
        "wordlist_printed_pages": "41-95",
        "counts": {
            "conceptual_list_slots": len(audit),
            "source_image_cells_manually_reviewed": len(seen_cells),
            "target_cells_manually_reviewed": 8 * 210,
            "comparison_cells_manually_reviewed": 210,
            "installed_rows": len(forms),
            "status": dict(status_counts),
            "uncertainty": dict(uncertainty_counts),
        },
        "policy": {
            "transcription": "manual visual review of every cell against the embedded source image",
            "ocr": "comparison scaffold only; never installed without manual source-image review",
            "controls": ["NIH"],
            "etymology": "none claimed; source similarity percentages are not cognacy",
        },
    }
    return forms, audit, metadata


def write(forms: list[list[str]], audit: list[dict], metadata: dict, install: bool) -> None:
    target = INSTALLED if install else REPO / "tmp/20260828-sil-korku-preview.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)
    MANIFEST.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"source_cells={len(audit)} installed={len(forms)} missing={sum(r['Status']=='missing' for r in audit)} controls={sum(r['Status']=='excluded' for r in audit)} output={target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    write(*build(), install=args.install)


if __name__ == "__main__":
    main()
