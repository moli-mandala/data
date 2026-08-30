#!/usr/bin/env python3
"""Install the image-only Appendix B wordlists of SIL ESR 2018-010.

The PDF is not redistributed.  ``tesseract_raw.txt`` is a reproducible structural
OCR pass; ``transcription.tsv`` is the checked, source-facing IPA review.  This
importer re-parses the OCR, verifies the review topology, and emits both the
installed Irula rows and a complete audit that accounts for target gaps,
comparison lists, and non-record OCR fragments.
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

from build_scaffold import parse, validate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
FORMS = REPO / "data/other/forms"
RAW_ROOT = FORMS / "raw_data"
RAW_OCR = HERE / "tesseract_raw.txt"
TRANSCRIPTION = HERE / "transcription.tsv"
INSTALLED = FORMS / "20260828-sil-nilgiri-irula.csv"
AUDIT = RAW_ROOT / "20260828-sil-nilgiri-irula-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-nilgiri-irula-manifest.json"

SOURCE_KEY = "ernest-oleary-kelsall2018irula"
SOURCE_PDF = "silesr2018_010.pdf"
SOURCE_SHA256 = "2e5a4ef0f4c941437d09a1c8fa49ba01d4fe79e0915ad9248ac7b83280fb4c62"
KEY_PREFIX = "silirula2018"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Report", "Raw_Line", "PDF_Page", "Printed_Page", "Column",
    "Gloss_Number", "Gloss", "Site_Code", "Site_Name", "Dialect_Group",
    "Comparison_Role", "Response", "Group", "Raw_OCR", "Transcription", "Review",
    "Uncertainty", "Status", "Reason", "Language_ID", "Dialect_ID", "Source",
    "Entry_Key",
]

# Appendix B site key; district/taluk assignments are printed in table 1 (p. 11).
SITES = {
    "KUN": ("Kunjapanai", "Mele Nadu", "target"),
    "KOL": ("Kolikarai", "Mele Nadu", "target"),
    "CHE": ("Chemmanarai", "Mele Nadu", "target"),
    "KIL": ("Kilkupkad", "Mele Nadu", "target"),
    "MET": ("Mettukal", "Mele Nadu", "target"),
    "CHO": ("Chokkanalli", "Northern", "target"),
    "MAV": ("Mavanalla", "Northern", "target"),
    "ANA": ("Anaikatty", "Northern", "target"),
    "BOO": ("Bookapuram", "Northern", "target"),
    "THA": ("Thaliyur", "Vette Kada", "target"),
    "NEL": ("Nellithurai", "Vette Kada", "target"),
    "CBT": ("Coimbatore Tamil", "", "elicited comparison list"),
    # The report never expands MAD.  Its Tamil-like forms are not enough to
    # assign it to a language, so its identity remains deliberately unresolved.
    "MAD": ("MAD (unresolved)", "", "unresolved comparison list"),
    "KAN": ("Kannada", "", "comparison list culled from Blair 2012"),
    "BAD": ("Badaga", "", "comparison list culled from Blair 2012"),
    "ALU": ("Alu Kurumba", "", "comparison list culled from Blair 2012"),
    "BET": ("Betta Kurumba", "", "comparison list culled from Blair 2012"),
    "JEN": ("Jenu Kurumba", "", "comparison list culled from Blair 2012"),
}

HEADER_CONTINUATIONS = {
    "wood)", "sized)", "red, dry)", "evening/afternoon", "informal )",
    "masculine)", "feminine)", "inclusive)", "exclusive)",
}


def slug(value: str) -> str:
    return "-".join("".join(c if c.isalnum() else " " for c in value.lower()).split())


def dialect_id(site_code: str) -> str:
    return f"nilgiri-irula-{slug(SITES[site_code][0])}"


def dialect_tag(site_code: str) -> str:
    sid = dialect_id(site_code)
    name = SITES[site_code][0]
    return f"dialect:Irula:{quote(sid, safe='')}:{quote(name, safe='')}"


def read_reviews() -> dict[tuple[int, str, int], dict[str, str]]:
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    reviews: dict[tuple[int, str, int], dict[str, str]] = {}
    for row in rows:
        key = (int(row["Item"]), row["Site"], int(row["Response"]))
        if key in reviews:
            raise AssertionError(f"duplicate transcription key {key}")
        if row["Review"] not in {"reviewed", "missing"}:
            raise AssertionError(f"unfinished transcription at {key}: {row['Review']}")
        if row["Review"] == "reviewed" and not row["Transcription"]:
            raise AssertionError(f"empty reviewed transcription at {key}")
        if row["Review"] == "missing" and row["Transcription"]:
            raise AssertionError(f"source-marked gap was filled at {key}")
        if unicodedata.normalize("NFC", row["Transcription"]) != row["Transcription"]:
            raise AssertionError(f"non-NFC transcription at {key}")
        reviews[key] = row
    return reviews


def source_locator(record: dict, site_name: str) -> str:
    return (
        f"{SOURCE_KEY}[Appendix B, printed p. {record['printed_page']}, "
        f"item {record['item']}, {site_name} ({record['site']})]"
    )


def audit_base(record: dict) -> dict[str, str | int]:
    site_name, dialect_group, role = SITES[record["site"]]
    return {
        "Record_Type": "wordlist response",
        "Report": SOURCE_KEY,
        "Raw_Line": record["line"],
        "PDF_Page": record["pdf_page"],
        "Printed_Page": record["printed_page"],
        "Column": record["column"],
        "Gloss_Number": record["item"],
        "Gloss": record["gloss"],
        "Site_Code": record["site"],
        "Site_Name": site_name,
        "Dialect_Group": dialect_group,
        "Comparison_Role": role,
        "Response": record["response"],
        "Group": record["group"],
        "Raw_OCR": record["ocr"],
        "Transcription": "",
        "Review": "not transcribed (excluded comparison)",
        "Uncertainty": "",
        "Status": "excluded",
        "Reason": "",
        "Language_ID": "",
        "Dialect_ID": "",
        "Source": source_locator(record, site_name),
        "Entry_Key": "",
    }


def build() -> tuple[list[dict], list[dict], dict]:
    records, fragments = parse(RAW_OCR)
    errors = validate(records)
    if errors:
        raise AssertionError("; ".join(errors))
    if len(records) != 3388 or len(fragments) != 29:
        raise AssertionError(f"source topology drift: {len(records)} records, {len(fragments)} fragments")

    reviews = read_reviews()
    target_keys = {
        (record["item"], record["site"], record["response"])
        for record in records if record["target"]
    }
    if set(reviews) != target_keys:
        missing = sorted(target_keys - set(reviews))[:10]
        extra = sorted(set(reviews) - target_keys)[:10]
        raise AssertionError(f"transcription/scaffold mismatch; missing={missing}, extra={extra}")

    installed: list[dict] = []
    audit: list[dict] = []
    for record in records:
        entry = audit_base(record)
        if not record["target"]:
            role = SITES[record["site"]][2]
            entry["Reason"] = f"excluded {role}; not a target Irula list"
            if record["site"] == "MAD":
                entry["Reason"] += "; the report does not expand the code MAD"
            audit.append(entry)
            continue

        review = reviews[(record["item"], record["site"], record["response"])]
        entry.update(
            {
                "Transcription": review["Transcription"],
                "Review": review["Review"],
                "Uncertainty": review["Uncertainty"],
                "Language_ID": "Irula",
                "Dialect_ID": dialect_id(record["site"]),
            }
        )
        if review["Review"] == "missing":
            entry["Reason"] = "source explicitly prints missing data"
            audit.append(entry)
            continue

        key = (
            f"{KEY_PREFIX}:g{record['item']:03d}:{dialect_id(record['site'])}:"
            f"i{record['response']}"
        )
        form = review["Transcription"]
        notes = (
            f"Appendix B lexical-similarity group {record['group']}; "
            f"{SITES[record['site']][1]} Irula"
        )
        if review["Uncertainty"]:
            notes += f"; source-raster review flags: {review['Uncertainty']}"
        row = {
            "Language_ID": "Irula",
            "Parameter_ID": "",
            "Form": form,
            "Gloss": record["gloss"],
            "Native": "",
            "Phonemic": form,
            "Notes": notes,
            "Source": entry["Source"],
            "Cognateset": "",
            "Etymology": "",
            "Entry_Key": key,
            "Variant_Of_Key": "",
            "Borrowed_From_Key": "",
            "Derivation_Parent_Keys": "",
            "Tags": dialect_tag(record["site"]),
        }
        installed.append(row)
        entry.update({"Status": "installed", "Reason": "", "Entry_Key": key})
        audit.append(entry)

    for fragment in fragments:
        raw = fragment["raw"].strip()
        continuation = raw in HEADER_CONTINUATIONS
        audit.append(
            {
                "Record_Type": "layout fragment",
                "Report": SOURCE_KEY,
                "Raw_Line": fragment["line"],
                "PDF_Page": fragment.get("pdf_page", ""),
                "Printed_Page": fragment.get("printed_page", ""),
                "Column": fragment.get("column", ""),
                "Gloss_Number": fragment.get("item", ""),
                "Gloss": "",
                "Site_Code": "",
                "Site_Name": "",
                "Dialect_Group": "",
                "Comparison_Role": "",
                "Response": "",
                "Group": "",
                "Raw_OCR": raw,
                "Transcription": "",
                "Review": "classified",
                "Uncertainty": "",
                "Status": "excluded",
                "Reason": (
                    "wrapped English gloss/header continuation; represented in glosses.tsv"
                    if continuation else
                    "page-edge or adjacent-column OCR artifact; not a wordlist response"
                ),
                "Language_ID": "",
                "Dialect_ID": "",
                "Source": (
                    f"{SOURCE_KEY}[Appendix B, printed p. {fragment.get('printed_page', '')}, "
                    f"OCR line {fragment['line']}]"
                ),
                "Entry_Key": "",
            }
        )

    keys = [row["Entry_Key"] for row in installed]
    if len(keys) != len(set(keys)):
        raise AssertionError("installed Entry_Key values are not unique")
    statuses = Counter(row["Status"] for row in audit)
    uncertainties = Counter(
        flag
        for row in audit
        for flag in str(row["Uncertainty"]).split("; ")
        if flag
    )
    manifest = {
        "source": SOURCE_KEY,
        "source_archive": 76656,
        "source_pdf": SOURCE_PDF,
        "source_pdf_sha256": SOURCE_SHA256,
        "scope": "Appendix B, printed pages 25-48, all 187 prompts and 18 lists",
        "installed_file": str(INSTALLED.relative_to(REPO)),
        "audit_file": str(AUDIT.relative_to(REPO)),
        "counts": {
            "parsed_wordlist_responses": len(records),
            "layout_fragments": len(fragments),
            "audit_rows": len(audit),
            "target_response_records": sum(record["target"] for record in records),
            "installed_target_forms": len(installed),
            "target_source_gaps": sum(
                row["Review"] == "missing" for row in reviews.values()
            ),
            "excluded_comparison_records": sum(not record["target"] for record in records),
            "status": dict(sorted(statuses.items())),
            "uncertainty_flags": dict(sorted(uncertainties.items())),
        },
        "target_sites": {
            code: {"name": name, "dialect_group": group, "dialect_id": dialect_id(code)}
            for code, (name, group, role) in SITES.items() if role == "target"
        },
        "comparison_codes": {
            code: {"name": name, "role": role}
            for code, (name, _group, role) in SITES.items() if role != "target"
        },
        "transcription_policy": [
            "IPA was manually reviewed against enlarged source raster crops; OCR supplied structure only",
            "The overlapping 2015 Palakkad Kunjapana list was independent review evidence, never substitution",
            "Low-resolution distinctions that cannot be guaranteed are preserved as typed uncertainty flags",
            "The source's 15 printed missing-data target records remain excluded gaps",
            "No cognate or etymological relationships are inferred from lexical-similarity groups",
        ],
        "validation": {
            "all_187_prompts": True,
            "all_18_site_codes_per_prompt_in_source_order": True,
            "unparsed_records": 0,
            "unicode_nfc": True,
            "unique_entry_keys": True,
        },
    }
    return installed, audit, manifest


def write(rows: list[dict], audit: list[dict], manifest: dict) -> None:
    with INSTALLED.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows([[row[field] for field in FORM_FIELDS] for row in rows])
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    rows, audit, manifest = build()
    counts = manifest["counts"]
    print(
        f"installed={len(rows)} target_gaps={counts['target_source_gaps']} "
        f"controls={counts['excluded_comparison_records']} audit={len(audit)} "
        f"unparsed={manifest['validation']['unparsed_records']}"
    )
    if args.install:
        write(rows, audit, manifest)
        print(f"wrote {INSTALLED.relative_to(REPO)}, {AUDIT.relative_to(REPO)}, and manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
