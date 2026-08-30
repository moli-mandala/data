#!/usr/bin/env python3
"""Build the manually reviewed ESR 2012-016 Konda Dora source package."""

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
DATA_ROOT = HERE.parents[4]
REVIEWED = HERE / "reviewed_transcription.psv"
OCR_SCAFFOLD = HERE / "ocr_scaffold.txt"
OUTPUT = DATA_ROOT / "data/other/forms/20260828-sil-konda-dora.csv"
AUDIT = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-konda-dora-audit.csv"
MANIFEST = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-konda-dora-manifest.json"
SOURCE_KEY = "blair-george2012kondadora"
PDF_SHA256 = "6e0a3e5522a45752938f8279753d07b4e29d7b76ca73e88f71c4e283dfd0f533"
PDF_URL = "https://www.sil.org/system/files/reapdata/38/76/91/38769117428458388974018399323322688545/silesr2012_016.pdf"
ARCHIVE_URL = "https://www.sil.org/resources/archives/49120"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Source_Key", "Prompt_Key", "Item", "Gloss", "List", "Role", "PDF_Page",
    "Printed_Page", "Similarity_Group", "Source_Cell", "Manual_Form",
    "Record_Type", "Review_Method", "Review_Status", "Confidence", "Status",
    "Reason", "Installed_Count", "Expanded_Forms", "Entry_Keys", "Language_ID",
    "Dialect_ID", "Citation", "Review_Note",
]

LISTS = {
    "Koraput": {
        "code": "K", "role": "target", "language": "Konda",
        "dialect": "sil-konda-dora-1987-koraput", "display": "Koraput Konda (Pansawalsa)",
    },
    "Visakh": {
        "code": "V", "role": "target", "language": "Konda",
        "dialect": "sil-konda-dora-1987-visakh", "display": "Visakh Konda (Lakshmipuram)",
    },
    "Telugu": {
        "code": "T", "role": "comparison control", "language": "", "dialect": "", "display": "Telugu",
    },
    "Adivasi_Oriya": {
        "code": "O", "role": "comparison control", "language": "", "dialect": "", "display": "Adivasi Oriya (Kotia Oriya)",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag(language: str, dialect: str, display: str) -> str:
    return f"dialect:{quote(language, safe='')}:{quote(dialect, safe='')}:{quote(display, safe='')}"


def parse_cell(source_cell: str) -> tuple[str, str, str]:
    """Return record type, similarity group, and diplomatic response string."""
    cell = unicodedata.normalize("NFC", source_cell.strip())
    if cell == "----":
        return "blank", "", ""
    if cell[:1].isdigit():
        return "response", cell[0], cell[1:]
    if cell.startswith("-"):
        return "response", "ungrouped", cell[1:]
    return "response", "", cell


def expand(item: int, form: str) -> list[tuple[str, str]]:
    parts = [part.strip() for part in form.split("/")]
    if 182 <= item <= 201:
        labels = ["third-person past", "imperative", "infinitive"]
        if len(parts) != 3:
            raise ValueError(f"verb item {item} must have three source slots: {form!r}")
        return [(part, labels[index]) for index, part in enumerate(parts) if part != "----"]
    return [(part, "source lexical alternative" if len(parts) > 1 else "single response")
            for part in parts if part != "----"]


def load_reviewed() -> list[dict[str, str]]:
    with REVIEWED.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    if len(rows) != 214:
        raise ValueError(f"expected 214 reviewed prompt rows, found {len(rows)}")
    required = {
        "Prompt_Key", "Item", "Gloss", "Target_PDF_Page", "Target_Printed_Page",
        "Control_PDF_Page", "Control_Printed_Page", *LISTS, "Review_Status",
        "Confidence", "Review_Note",
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError("reviewed_transcription.psv has the wrong schema")
    if len({row["Prompt_Key"] for row in rows}) != 214:
        raise ValueError("prompt keys are not unique")
    if {row["Review_Status"] for row in rows} != {"manually_verified"}:
        raise ValueError("every source row must have complete manual visual review")
    if {row["Confidence"] for row in rows} != {"high"}:
        raise ValueError("unresolved or non-high-confidence readings require explicit handling")
    return rows


def build(rows: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]], dict[str, int]]:
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for row in rows:
        item = int(row["Item"])
        for list_name, metadata in LISTS.items():
            is_target = metadata["role"] == "target"
            pdf_page = row["Target_PDF_Page"] if is_target else row["Control_PDF_Page"]
            printed_page = row["Target_Printed_Page"] if is_target else row["Control_Printed_Page"]
            record_type, group, manual_form = parse_cell(row[list_name])
            citation = (
                f"{SOURCE_KEY}[Appendix 9.5, PDF p. {pdf_page}, printed p. {printed_page}, "
                f"item {row['Item']} ({row['Gloss']}), list {metadata['code']}]"
            )
            installed: list[tuple[str, str]] = []
            keys: list[str] = []
            if record_type == "response" and is_target:
                installed = expand(item, manual_form)
                for part_number, (form, function) in enumerate(installed, 1):
                    entry_key = (
                        f"silkondadora2012:{row['Prompt_Key']}:{metadata['code']}:"
                        f"{part_number}"
                    )
                    keys.append(entry_key)
                    notes = f"source lexical-similarity group {group}; {function}"
                    forms.append([
                        metadata["language"], "", form, row["Gloss"], "", form,
                        notes, citation, "", "", entry_key, "", "", "",
                        tag(metadata["language"], metadata["dialect"], metadata["display"]),
                    ])

            if record_type == "blank":
                status, reason = "excluded", "source prints ‘----’ (no response)"
            elif not is_target:
                status, reason = "excluded", f"{metadata['display']} comparison control"
            else:
                status, reason = "installed", ""
            audit.append(dict(zip(AUDIT_FIELDS, [
                SOURCE_KEY, row["Prompt_Key"], row["Item"], row["Gloss"], list_name,
                metadata["role"], pdf_page, printed_page, group, row[list_name],
                manual_form, record_type,
                "manual cell-by-cell visual transcription from rendered canonical scan; OCR/text layer used only as a locating scaffold",
                row["Review_Status"], row["Confidence"], status, reason,
                str(len(installed)), " | ".join(form for form, _ in installed),
                " | ".join(keys), metadata["language"] if is_target else "",
                metadata["dialect"] if is_target else "", citation, row["Review_Note"],
            ])))

    if len(audit) != 856:
        raise AssertionError(f"expected 856 conceptual cells, built {len(audit)}")
    counter = Counter(record["Status"] for record in audit)
    counts = {
        "prompts": len(rows),
        "lists": len(LISTS),
        "conceptual_source_cells_manually_reviewed": len(audit),
        "target_cells_manually_reviewed": sum(record["Role"] == "target" for record in audit),
        "control_cells_manually_reviewed": sum(record["Role"] != "target" for record in audit),
        "attested_cells": sum(record["Record_Type"] == "response" for record in audit),
        "confirmed_blank_cells": sum(record["Record_Type"] == "blank" for record in audit),
        "confirmed_blank_target_cells": sum(record["Record_Type"] == "blank" and record["Role"] == "target" for record in audit),
        "confirmed_blank_control_cells": sum(record["Record_Type"] == "blank" and record["Role"] != "target" for record in audit),
        "excluded_control_response_cells": sum(record["Record_Type"] == "response" and record["Role"] != "target" for record in audit),
        "installed_forms_after_source_defined_expansion": len(forms),
        "audit_rows": len(audit),
        "unresolved_or_illegible_cells": 0,
        "source_marked_uncertain_cells": 0,
    }
    if counter["installed"] != sum(record["Role"] == "target" and record["Record_Type"] == "response" for record in audit):
        raise AssertionError("installed cell count disagrees with target response count")
    return forms, audit, counts


def write_forms(rows: list[list[str]]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def write_audit(rows: list[dict[str, str]]) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows = load_reviewed()
    forms, audit, counts = build(rows)
    if args.install:
        write_forms(forms)
        write_audit(audit)
        manifest = {
            "source_key": SOURCE_KEY,
            "title": "Multilingualism Among the Konda Dora",
            "authors": ["Frank Blair", "Jacob George"],
            "researchers": ["Susan George", "Stephen Watters"],
            "year": 2012,
            "created": 1987,
            "series": "SIL Electronic Survey Reports 2012-016",
            "archive_url": ARCHIVE_URL,
            "pdf_url": PDF_URL,
            "pdf_sha256": PDF_SHA256,
            "pdf_bytes": 31978201,
            "pdf_pages": 106,
            "scope": "Appendix 9.5, PDF pp. 88-106, printed pp. 83-101; lexical tables PDF pp. 89-106",
            "review": {
                "authority": "rendered canonical scan",
                "manual": "all 856 target and control cells were visually inspected and manually transcribed; every installed expansion was compared against its rendered source cell",
                "ocr": "the PDF text layer is retained only as a locating/comparison scaffold and never supplies an accepted reading",
                "unresolved": [],
                "transcription_decisions": [
                    "Similarity-group digits and a leading source hyphen are structural and are excluded from lexical Form while retained in the audit.",
                    "Typewriter underdots are represented with NFC Unicode retroflex characters where available.",
                    "The visibly distinct source IPA retroflex-flap glyph is retained as ɽ and converted to Jambu display ṛ by the profile.",
                    "The literal source question-mark glyph is retained diplomatically in Form; the source-local profile maps it to glottal stop for conversion.",
                    "Items 182-201 are expanded by the source-defined past/imperative/infinitive slots; other slashes are source-defined lexical alternatives.",
                    "The duplicate printed item number 212 is retained as distinct 212-liver and 212-foot prompt keys.",
                ],
            },
            "counts": counts,
            "artifacts": {
                "reviewed_transcription": {"path": str(REVIEWED.relative_to(DATA_ROOT)), "sha256": sha256(REVIEWED)},
                "ocr_scaffold": {"path": str(OCR_SCAFFOLD.relative_to(DATA_ROOT)), "sha256": sha256(OCR_SCAFFOLD), "authority": "none; locator/comparison only"},
                "installed": {"path": str(OUTPUT.relative_to(DATA_ROOT)), "sha256": sha256(OUTPUT)},
                "audit": {"path": str(AUDIT.relative_to(DATA_ROOT)), "sha256": sha256(AUDIT)},
            },
        }
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
