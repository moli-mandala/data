#!/usr/bin/env python3
"""Build the source-local JLSR 2022-004 Bonda/Didayi wordlist package."""

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
REPO = HERE.parents[4]
CELLS = HERE / "extracted_cells.tsv"
PDF = REPO.parent / "tmp/pdfs/bonda_didayi/JLSR2022_004.pdf"
FORMS = REPO / "data/other/forms/20260828-sil-bonda-didayi.csv"
AUDIT = REPO / "data/other/forms/raw_data/20260828-sil-bonda-didayi-audit.csv"
MANIFEST = REPO / "data/other/forms/raw_data/20260828-sil-bonda-didayi-manifest.json"

SOURCE_KEY = "mathew-chamberlain2022bonda-didayi"
SOURCE_SHA256 = "bb0548b4324224260b9618786dfd3aa40377138d0fbf4ae14c796df82f6190ce"
KEY_PREFIX = "silbondadidayi1997"

# code: source label, role, base language ID
SITES = {
    "BIA": ("Biapada U. Didayi", "target", "gt"),
    "CHI": ("Chitrakonda L. Didayi", "target", "gt"),
    "KAL": ("Kaluguda U. Didayi", "target", "gt"),
    "ORA": ("Orapadar U. Didayi", "target", "gt"),
    "ORI": ("Oringi L. Didayi", "target", "gt"),
    "RAS": ("Rasabeda L. Bonda", "target", "re"),
    "KEN": ("Kendhuguda L. Bonda", "target", "re"),
    "KAD": ("Kadamguda L. Bonda", "target", "re"),
    "DUM": ("Dumripada U. Bonda", "target", "re"),
    "GUT": ("Tikrapada Gutob", "comparison control", "gu"),
    "PAR": ("Kinumun Parenga Parja", "comparison control", "go"),
    "RON": ("Malenga Rona Desiya", "comparison control", "AdivasiOriya"),
    "ODI": ("Cuttack Oriya", "comparison control", "Or"),
}
TARGETS = {code for code, (_, role, _) in SITES.items() if role == "target"}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Item", "Gloss", "Site_Code", "Site_Name", "Role",
    "PDF_Page", "Printed_Page", "Column", "Raw_Response", "Similarity_Groups",
    "Reviewed_Transcription", "Manual_Review", "Uncertainty", "Status", "Reason",
    "Language_ID", "Dialect_ID", "Source", "Entry_Keys",
]


def slug(value: str) -> str:
    return "-".join("".join(c if c.isalnum() else " " for c in value.lower()).split())


def dialect_id(code: str) -> str:
    return f"sil-bonda-didayi-1997-{slug(SITES[code][0])}"


def dialect_tag(code: str) -> str:
    name, _, language = SITES[code]
    return f"dialect:{language}:{quote(dialect_id(code), safe='')}:{quote(name, safe='')}"


def locator(row: dict[str, str]) -> str:
    return (
        f"{SOURCE_KEY}[Appendix B, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, {row['Site_Name']}]"
    )


def clean_layout(value: str) -> str:
    """Remove line-layout spacing but preserve source punctuation and IPA."""
    value = unicodedata.normalize("NFC", value.strip())
    value = re.sub(r"-\s+", "-", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_response(raw: str) -> tuple[list[tuple[str, str]], str]:
    """Return source comma-separated forms with their printed similarity groups."""
    raw = clean_layout(raw)
    if not raw or raw == "DISQUALIFIED" or raw == "---":
        return [], ""
    pieces = [part.strip() for part in raw.rstrip(",").split(",")]
    parsed, inherited = [], ""
    for piece in pieces:
        match = re.match(r"^(\d+)\s*(.*)$", piece)
        if match:
            inherited, form = match.groups()
        else:
            form = piece
        form = clean_layout(form)
        compact = "".join(c for c in unicodedata.normalize("NFD", form).lower() if c.isalnum())
        if inherited == "0" or compact == "noentry":
            continue
        if form:
            parsed.append((inherited, form))
    return parsed, "|".join(group for group, _ in parsed)


def read_cells() -> list[dict[str, str]]:
    with CELLS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    coordinates = {(int(row["Item"]), row["Site_Code"]) for row in rows}
    expected = {(item, code) for item in range(1, 211) for code in SITES}
    if len(rows) != 2730 or coordinates != expected:
        raise AssertionError("review ledger topology drift")
    if any(unicodedata.normalize("NFC", row["Raw_Response"]) != row["Raw_Response"] for row in rows):
        raise AssertionError("review ledger must be NFC")
    return rows


def build() -> tuple[list[list[str]], list[dict[str, str]], dict]:
    if hashlib.sha256(PDF.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise AssertionError("canonical PDF checksum drift")
    forms, audit = [], []
    for row in read_cells():
        item, code = int(row["Item"]), row["Site_Code"]
        name, role, language = SITES[code]
        if name != row["Site_Name"]:
            raise AssertionError(f"site label drift at {(item, code)}")
        source = locator(row)
        parsed, groups = parse_response(row["Raw_Response"])
        entry = {
            "Record_Type": "wordlist cell", "Item": str(item), "Gloss": row["Gloss"],
            "Site_Code": code, "Site_Name": name, "Role": role,
            "PDF_Page": row["PDF_Page"], "Printed_Page": row["Printed_Page"],
            "Column": row["Column"], "Raw_Response": row["Raw_Response"],
            "Similarity_Groups": groups,
            "Reviewed_Transcription": " | ".join(form for _, form in parsed),
            "Manual_Review": "visual-source-page", "Uncertainty": "", "Status": "",
            "Reason": "", "Language_ID": language,
            "Dialect_ID": dialect_id(code) if code in TARGETS else "",
            "Source": source, "Entry_Keys": "",
        }
        if role != "target":
            reason = "excluded comparison control"
            if row["Extraction_Status"] == "disqualified":
                reason += "; prompt disqualified by source"
            entry.update(Status="excluded", Reason=reason)
            audit.append(entry)
            continue
        if row["Extraction_Status"] == "disqualified":
            entry.update(Status="disqualified", Reason="prompt disqualified by source")
            audit.append(entry)
            continue
        if row["Extraction_Status"] == "source-omitted":
            entry.update(
                Status="missing", Reason="source physically omits this locality row",
                Uncertainty="source omission; no form printed",
            )
            audit.append(entry)
            continue
        if not parsed:
            marker = clean_layout(row["Raw_Response"])
            entry.update(Status="missing", Reason=f"source prints no response ({marker})")
            audit.append(entry)
            continue
        keys = []
        for variant, (group, form) in enumerate(parsed, 1):
            key = f"{KEY_PREFIX}:{code.lower()}:i{item:03d}:v{variant}"
            keys.append(key)
            note = "Appendix B; visually verified born-digital source transcription"
            if group:
                note += f"; source similarity group {group} (descriptive only)"
            forms.append([
                language, "", form, row["Gloss"], "", form, note, source, "", "", key,
                "", "", "", dialect_tag(code),
            ])
        entry.update(Status="installed", Entry_Keys="|".join(keys))
        audit.append(entry)
    audit.sort(key=lambda row: (int(row["Item"]), list(SITES).index(row["Site_Code"])))
    counts = Counter(row["Status"] for row in audit)
    metadata = {
        "source_key": SOURCE_KEY,
        "source_pdf_sha256": SOURCE_SHA256,
        "extracted_cells_sha256": hashlib.sha256(CELLS.read_bytes()).hexdigest(),
        "source_archive_entry": 92608,
        "source_pdf_pages": 64,
        "wordlist_pdf_pages": "21-50",
        "wordlist_printed_pages": "16-45",
        "counts": {
            "conceptual_cells_visually_reviewed": len(audit),
            "target_cells_visually_reviewed": sum(row["Site_Code"] in TARGETS for row in audit),
            "comparison_cells_visually_reviewed": sum(row["Site_Code"] not in TARGETS for row in audit),
            "installed_rows": len(forms), "status": dict(counts),
        },
        "unresolved": [{"pdf_page": 45, "printed_page": 40, "item": 174, "site": "ORA",
                        "reason": "source physically omits the Orapadar row"}],
        "policy": {
            "review": "every cell visually compared against rendered Appendix B pages 21-50",
            "ocr": "not used; appendix is born-digital Unicode",
            "controls": ["GUT", "PAR", "RON", "ODI"],
            "similarity": "printed similarity numbers retained only as descriptive notes; no cognacy inferred",
        },
    }
    return forms, audit, metadata


def write(forms: list[list[str]], audit: list[dict[str, str]], metadata: dict, install: bool) -> None:
    target = FORMS if install else REPO / "tmp/20260828-sil-bonda-didayi-preview.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(forms)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader(); writer.writerows(audit)
    MANIFEST.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"cells={len(audit)} installed={len(forms)} status={metadata['counts']['status']} output={target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    write(*build(), install=args.install)


if __name__ == "__main__":
    main()
