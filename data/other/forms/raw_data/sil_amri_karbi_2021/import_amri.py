#!/usr/bin/env python3
"""Install visually reviewed JLSR 2021-050 Karbi and Amri Karbi wordlists."""

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
REVIEWED = HERE / "reviewed_transcription.tsv"
OUTPUT = DATA_ROOT / "data/other/forms/20260828-sil-amri-karbi.csv"
AUDIT = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-amri-karbi-audit.csv"
MANIFEST = DATA_ROOT / "data/other/forms/raw_data/20260828-sil-amri-karbi-manifest.json"
SOURCE_KEY = "abraham-daimary2021amrikarbi"
PDF_SHA256 = "cd121ad102e96b43bf68a1cc5b44f1559c764bc4ae8d71988c6b292a1896ccb1"
PDF_URL = "https://www.sil.org/system/files/reapdata/13/17/32/131732907821549471875367421669287002635/JLSR2021_050.pdf"
ARCHIVE_URL = "https://www.sil.org/resources/archives/91601"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Source_Key", "PDF_Page", "Printed_Page", "Page_Line", "Item", "Gloss",
    "Source_Code", "Site", "Source_Classification", "Similarity_Group",
    "Extracted_Form", "Verified_Form", "Review_Method", "Review_Status",
    "Confidence", "Record_Type", "Status", "Reason", "Language_ID",
    "Dialect_ID", "Citation", "Entry_Key",
]

# source site -> code, source classification, Jambu base language, dialect id,
# display name, locality description.  Assamese and Khasi are controls.
SITES = {
    "Holanki, Papumpare AP": ("A", "Karbi", "karbi", "sil-amri-karbi-2021-holanki", "Holanki", "Holanki, Papumpare district, Arunachal Pradesh"),
    "S Cherrapunjee": ("C", "Khasi control", "", "", "Sohra (Cherrapunjee)", "Sohra, East Khasi Hills district, Meghalaya"),
    "Hajarongpi, E K A": ("H", "Karbi", "karbi", "sil-amri-karbi-2021-hajarongpi", "Hajarongpi", "Hajarongpi, East Karbi Anglong district, Assam"),
    "Amguri, Kamrup": ("K", "Amri Karbi", "amri_karbi", "sil-amri-karbi-2021-amguri-kamrup", "Amguri (Kamrup)", "Amguri, Kamrup district, Assam"),
    "PaboiMisamari, Sonitpur": ("M", "Karbi", "karbi", "sil-amri-karbi-2021-paboi-misamari", "Paboi Misamari", "Paboi Misamari, Sonitpur district, Assam"),
    "Maina Kharong, Kamrup": ("P", "Amri Karbi", "amri_karbi", "sil-amri-karbi-2021-maina-kharong", "Maina Kharong", "Maina Kharong, Kamrup district, Assam"),
    "RongjariPlasha, Ri-Bhoi": ("S", "Amri Karbi", "amri_karbi", "sil-amri-karbi-2021-plasha", "Plasha (Rongjari)", "Rongjari Plasha, Ri-Bhoi district, Meghalaya"),
    "Assamese, Dibrugarh": ("Z", "Assamese control", "", "", "Assamese (Dibrugarh)", "Dibrugarh district, Assam"),
    "Amguri, W K A": ("a", "Karbi", "karbi", "sil-amri-karbi-2021-amguri-wka", "Amguri (West Karbi Anglong)", "Amguri, West Karbi Anglong district, Assam"),
    "Sermansingner, E K A": ("b", "Karbi", "karbi", "sil-amri-karbi-2021-sermansingner", "Sermansingner", "Sermansingner, East Karbi Anglong district, Assam"),
    "Langhemphi, W K A": ("c", "Karbi", "karbi", "sil-amri-karbi-2021-langhemphi", "Langhemphi", "Langhemphi, West Karbi Anglong district, Assam"),
    "Umrinti, W K A": ("d", "Karbi", "karbi", "sil-amri-karbi-2021-umrinti", "Umrinti", "Umrinti, West Karbi Anglong district, Assam"),
    "Bankri, W K A": ("h", "Karbi", "karbi", "sil-amri-karbi-2021-bankri", "Bankri (Bhankri)", "Bhankri, West Karbi Anglong district, Assam"),
    "Rongtheang, E K A": ("k", "Karbi", "karbi", "sil-amri-karbi-2021-rongtheang", "Rongtheang", "Rongtheang, East Karbi Anglong district, Assam"),
    "Sunajoli, Lakhimpur": ("l", "Karbi", "karbi", "sil-amri-karbi-2021-sunajoli", "Sunajoli", "Sunajoli, Lakhimpur district, Assam"),
    "Mikirgaon, Nagaon": ("m", "Karbi", "karbi", "sil-amri-karbi-2021-mikirgaon", "Mikirgaon", "Mikirgaon, Nagaon district, Assam"),
    "Sardoka Ingti, E K A": ("s", "Karbi", "karbi", "sil-amri-karbi-2021-sardoka-ingti", "Sardoka Ingti", "Sardoka Ingti, East Karbi Anglong district, Assam"),
}
CONTROL_CODES = {"C", "Z"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag(language_id: str, dialect_id: str, display: str) -> str:
    return f"dialect:{quote(language_id, safe='')}:{quote(dialect_id, safe='')}:{quote(display, safe='')}"


def load_reviewed() -> list[dict[str, str]]:
    with REVIEWED.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {
        "PDF_Page", "Printed_Page", "Page_Line", "Item", "Gloss", "Site",
        "Similarity_Group", "Extracted_Form", "Verified_Form", "Record_Type",
        "Review_Status", "Confidence", "Review_Note",
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError("reviewed_transcription.tsv has the wrong schema")
    if len(rows) != 5_966:
        raise ValueError(f"expected 5,966 reviewed source records, found {len(rows)}")
    allowed_statuses = {"complete", "source-marked-uncertain"}
    if any(row["Review_Status"] not in allowed_statuses for row in rows):
        raise ValueError("every source record must have complete visual review")
    if any(row["Site"] not in SITES for row in rows):
        raise ValueError("review ledger contains an unknown site")
    if any(row["Extracted_Form"] != row["Verified_Form"] for row in rows if row["Record_Type"] == "response"):
        raise ValueError("verified corrections must be explicit and documented before import")
    return rows


def build(rows: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]], dict[str, object]]:
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    seen_form: dict[tuple[str, str, str], str] = {}
    duplicate_occurrences = 0

    for index, row in enumerate(rows, 1):
        code, classification, language_id, dialect_id, display, _ = SITES[row["Site"]]
        target = code not in CONTROL_CODES
        form = unicodedata.normalize("NFC", row["Verified_Form"])
        citation = f"{SOURCE_KEY}[Appendix B.3, PDF p. {row['PDF_Page']}, printed p. {row['Printed_Page']}, item {row['Item']}, site {code}]"
        notes = f"source lexical-similarity group {row['Similarity_Group']}"
        entry_key = ""
        status = "excluded"
        reason = ""
        if row["Record_Type"] == "blank":
            reason = "source prints ‘no entry’"
        elif not target:
            reason = f"{classification}"
        else:
            identity = (row["Item"], code, form)
            if identity in seen_form:
                duplicate_occurrences += 1
                reason = f"exact repeated source occurrence; installed once as {seen_form[identity]}"
            else:
                entry_key = f"silamrikarbi2021:p{int(row['PDF_Page']):03d}:l{int(row['Page_Line']):03d}:{code}"
                seen_form[identity] = entry_key
                forms.append([
                    language_id, "", form, row["Gloss"], "", form, notes,
                    citation, "", "", entry_key, "", "", "",
                    tag(language_id, dialect_id, display),
                ])
                status = "installed"
        audit.append(dict(zip(AUDIT_FIELDS, [
            SOURCE_KEY, row["PDF_Page"], row["Printed_Page"], row["Page_Line"],
            row["Item"], row["Gloss"], code, row["Site"], classification,
            row["Similarity_Group"], row["Extracted_Form"], row["Verified_Form"],
            "manual visual comparison against rendered canonical PDF page; PDF text layer used only as extraction scaffold",
            row["Review_Status"], row["Confidence"], row["Record_Type"], status,
            reason, language_id if status == "installed" else "",
            dialect_id, citation, entry_key,
        ])))

    counts = Counter(record["Status"] for record in audit)
    by_language = Counter(row[0] for row in forms)
    conceptual_cells = {(row["Item"], row["Site"]) for row in rows}
    blank_cells = sum(row["Record_Type"] == "blank" for row in rows)
    control_occurrences = sum(SITES[row["Site"]][0] in CONTROL_CODES and row["Record_Type"] == "response" for row in rows)
    manifest_counts = {
        "prompts": 307,
        "printed_lists": 17,
        "published_wordlists_reported": 21,
        "published_lists_absent_from_appendix_b3": 4,
        "conceptual_source_cells_manually_reviewed": len(conceptual_cells),
        "printed_response_occurrences_manually_reviewed": sum(row["Record_Type"] == "response" for row in rows),
        "confirmed_blank_cells": blank_cells,
        "target_printed_response_occurrences": sum(SITES[row["Site"]][0] not in CONTROL_CODES and row["Record_Type"] == "response" for row in rows),
        "excluded_control_response_occurrences": control_occurrences,
        "duplicate_target_occurrences_audit_only": duplicate_occurrences,
        "installed_forms": len(forms),
        "installed_amri_karbi_forms": by_language["amri_karbi"],
        "installed_karbi_forms": by_language["karbi"],
        "audit_rows": len(audit),
        "source_marked_uncertain_readings": sum(row["Review_Status"] == "source-marked-uncertain" for row in rows),
        "unresolved_transcriptions": sum(row["Confidence"] != "high" for row in rows),
    }
    assert counts["installed"] == len(forms)
    return forms, audit, manifest_counts


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
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
        write_csv(OUTPUT, forms)
        write_audit(AUDIT, audit)
        manifest = {
            "source_key": SOURCE_KEY,
            "title": "A Sociolinguistic Study of Amri Karbi [ajz] in Northeast India",
            "authors": ["Binny Abraham", "Pronay Daimary"],
            "year": 2021,
            "series": "Journal of Language Survey Reports 2021-050",
            "archive_url": ARCHIVE_URL,
            "pdf_url": PDF_URL,
            "pdf_sha256": PDF_SHA256,
            "pdf_pages": 165,
            "scope": "Appendix B.3, PDF pp. 37-115, printed pp. 27-105",
            "review": {
                "authority": "rendered canonical PDF pages",
                "text_layer": "extraction scaffold only; every installed form visually compared",
                "ocr": "not used; appendix has a structured Unicode text layer",
                "unresolved": [],
                "source_marked_uncertainty": [
                    {
                        "pdf_page": 59,
                        "printed_page": 49,
                        "item": 91,
                        "site": "Z (Assamese, Dibrugarh; excluded control)",
                        "reading": "soʌ̆ĭ??",
                        "note": "Two literal question marks are visible in the source and retained exactly; the transcription itself is visually secure.",
                    }
                ],
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
