#!/usr/bin/env python3
"""Install the 22 newly collected wordlists in SIL ESR 2019-006."""

from __future__ import annotations

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
RAW_ROOT = FORMS / "raw_data"
SNAPSHOT = HERE / "wordlist_snapshot.tsv"
INSTALLED = FORMS / "20260828-sil-lahul.csv"
AUDIT = RAW_ROOT / "20260828-sil-lahul-audit.csv"
MANIFEST = RAW_ROOT / "20260828-sil-lahul-manifest.json"

SOURCE_KEY = "chamberlain-chamberlain2019lahul"
SOURCE_PDF = "silesr2019_006.pdf"
SOURCE_SHA256 = "17f8178505ef88879baecbd5d9fa6dd4f2bb885330722cbac21df70c71e47252"
SNAPSHOT_SHA256 = "b0e83a40b929288d7b59a6ba3789096648a954b6ddab33ed6d6c182d5afcc963"
ENTRY_PREFIX = "sillahul2019"

LANGUAGES = {
    "Ch": ("cih", "Chinali", "chin1475"),
    "Lo": ("lhl", "Lahul Lohar", "lahu1250"),
    "Pa": ("lae", "Pattani", "patt1248"),
    "Ti": ("lbf", "Tinani", "tina1246"),
    "Ga": ("bfu", "Bunan (Gahri)", "gahr1239"),
    "Bh": ("sbu", "Stod Bhoti", "stod1241"),
}
TAG_LANGUAGE_LABELS = {
    "Ch": "Chinali", "Lo": "Lahul-Lohar", "Pa": "Pattani",
    "Ti": "Tinani", "Ga": "Gahri", "Bh": "Bhoti",
}

# Source sites are locality labels, not Glottolog dialect claims. Coordinates
# are modern gazetteer points and are registered at quality C in dialects.csv.
SITE_SPECS = {
    ("Ch", "Gushal"): (32.55208, 76.97081, "GeoNames 1270599"),
    ("Ch", "Nalda"): (32.6398722, 76.8334481, "OpenStreetMap node 8623987754"),
    ("Lo", "Gondhla"): (32.5132678, 77.0146942, "OpenStreetMap node 342116481"),
    ("Lo", "Gawzang"): (32.5590450, 77.0053078, "OpenStreetMap node 6319287361; modern Gotsang"),
    ("Pa", "Jobrang"): (32.6240537, 76.8715040, "OpenStreetMap node 8624123541"),
    ("Pa", "Thirot"): (32.65878, 76.78328, "GeoNames 1254591"),
    ("Pa", "Udeypur"): (32.7258662, 76.6652087, "OpenStreetMap node 10224069653; modern Udaipur"),
    ("Pa", "Gushal"): (32.55208, 76.97081, "GeoNames 1270599"),
    ("Pa", "Mooling"): (32.5133, 76.9757, "Himachal Pradesh Forest Department locality table"),
    ("Pa", "Tholang"): (32.5741479, 76.9581920, "OpenStreetMap node 7165541960"),
    ("Pa", "Chimrat"): (32.7887048, 76.7211368, "OpenStreetMap node 926152166; modern Chamrat"),
    ("Pa", "Salgram"): (32.75959, 76.54347, "GeoNames 1257618; modern Salugran/Salgaraon"),
    ("Ti", "Sissu"): (32.47391, 77.12998, "GeoNames 1255997"),
    ("Ti", "Gondhla"): (32.5132678, 77.0146942, "OpenStreetMap node 342116481"),
    ("Ga", "Keylong"): (32.5717891, 77.0281479, "OpenStreetMap node 342115051"),
    ("Ga", "Stingri"): (32.5604849, 77.0718077, "OpenStreetMap node 342116494"),
    ("Ga", "Gawzang"): (32.5590450, 77.0053078, "OpenStreetMap node 6319287361; modern Gotsang"),
    ("Bh", "Darcha"): (32.67324, 77.21431, "GeoNames 1273488"),
    ("Bh", "Kolong"): (32.5954416, 77.1373533, "OpenStreetMap node 14016635377"),
    ("Bh", "Rarig"): (32.7115101, 77.1689885, "OpenStreetMap node 14050516616; modern Rarik"),
    ("Bh", "Tingrat"): (32.8544487, 76.7896588, "OpenStreetMap node 926103055"),
    ("Bh", "Khoksar"): (32.40874, 77.2519, "GeoNames 1266679"),
}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Record_Type", "Source_Key", "PDF_Page", "Printed_Page", "Item", "Gloss",
    "Lect_Code", "Language_Label", "Site", "Similarity_Group", "Response_Index",
    "Raw_Form", "Form", "Review", "Uncertainty", "Source_Scope", "Status",
    "Reason", "Language_ID", "Dialect_ID", "Citation", "Entry_Key",
]


def dialect_id(code: str, site: str) -> str:
    return f"sil-lahul-2019-{code.lower()}-{site.lower().replace(' ', '-')}"


def dialect_name(code: str, site: str) -> str:
    return f"{site} ({LANGUAGES[code][1]})"


def dialect_tag(code: str, site: str) -> str:
    language_id = LANGUAGES[code][0]
    did = dialect_id(code, site)
    return (
        f"dialect:{language_id}:{quote(did, safe='')}:"
        f"{site}-{TAG_LANGUAGE_LABELS[code]}"
    )


def locator(row: dict[str, str]) -> str:
    return (
        f"{SOURCE_KEY}[Appendix A.4, printed p. {row['Printed_Page']}, "
        f"item {row['Item']}, {row['Language_Label']}-{row['Site']}]"
    )


def read_snapshot() -> list[dict[str, str]]:
    if hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() != SNAPSHOT_SHA256:
        raise AssertionError("frozen Lahul snapshot fingerprint drift")
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 6206:
        raise AssertionError(f"source response-count drift: {len(rows)}")
    return rows


def build() -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    source_rows = read_snapshot()
    target_pairs = {
        (row["Lect_Code"], row["Site"])
        for row in source_rows if row["Source_Scope"] == "target"
    }
    if target_pairs != set(SITE_SPECS):
        raise AssertionError("target site topology drift")

    installed: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    for row in source_rows:
        target = row["Source_Scope"] == "target"
        lexical = row["Status"] == "response"
        install = target and lexical
        code, site = row["Lect_Code"], row["Site"]
        language_id = LANGUAGES[code][0] if target else ""
        did = dialect_id(code, site) if target else ""
        citation = locator(row)
        entry_key = ""
        if install:
            entry_key = (
                f"{ENTRY_PREFIX}:g{int(row['Item']):03d}:{did}:"
                f"i{int(row['Response_Index'])}"
            )
            notes = (
                f"Appendix A.4 lexical-similarity group {row['Similarity_Group']}; "
                "source field transcription is explicitly not a phonological analysis"
            )
            installed.append({
                "Language_ID": language_id,
                "Parameter_ID": "",
                "Form": row["Form"],
                "Gloss": row["Gloss"],
                "Native": "",
                "Phonemic": row["Form"],
                "Notes": notes,
                "Source": citation,
                "Cognateset": "",
                "Etymology": "",
                "Entry_Key": entry_key,
                "Variant_Of_Key": "",
                "Borrowed_From_Key": "",
                "Derivation_Parent_Keys": "",
                "Tags": dialect_tag(code, site),
            })

        if install:
            reason = ""
            audit_status = "installed"
        elif target:
            reason = "source prints no entry rather than a lexical response"
            audit_status = "excluded"
        elif row["Source_Scope"] == "prior_list":
            reason = "previously collected regional comparison list; retained audit-only"
            audit_status = "excluded"
        else:
            reason = "standard comparison/control language; retained audit-only"
            audit_status = "excluded"
        audit.append({
            "Record_Type": "wordlist response",
            "Source_Key": SOURCE_KEY,
            "PDF_Page": row["PDF_Page"],
            "Printed_Page": row["Printed_Page"],
            "Item": row["Item"],
            "Gloss": row["Gloss"],
            "Lect_Code": code,
            "Language_Label": row["Language_Label"],
            "Site": site,
            "Similarity_Group": row["Similarity_Group"],
            "Response_Index": row["Response_Index"],
            "Raw_Form": row["Raw_Form"],
            "Form": row["Form"],
            "Review": row["Review"],
            "Uncertainty": "",
            "Source_Scope": row["Source_Scope"],
            "Status": audit_status,
            "Reason": reason,
            "Language_ID": language_id,
            "Dialect_ID": did,
            "Citation": citation,
            "Entry_Key": entry_key,
        })

    if len(installed) != 5027 or len(audit) != 6206:
        raise AssertionError(
            f"installation topology drift: installed={len(installed)} audit={len(audit)}"
        )
    if Counter(row["Language_ID"] for row in installed) != Counter({
        "cih": 427, "lhl": 427, "lae": 1807,
        "lbf": 441, "bfu": 668, "sbu": 1257,
    }):
        raise AssertionError("per-language installation topology drift")
    if len({row["Entry_Key"] for row in installed}) != len(installed):
        raise AssertionError("duplicate installed Entry_Key")
    if any(not row["Form"] or row["Form"] != row["Phonemic"] for row in installed):
        raise AssertionError("blank or mismatched installed transcription")
    if any(unicodedata.normalize("NFC", row["Form"]) != row["Form"] for row in installed):
        raise AssertionError("non-NFC installed form")
    if any(row["Parameter_ID"] or row["Cognateset"] for row in installed):
        raise AssertionError("survey similarity groups must not become etymological claims")

    manifest = {
        "source_key": SOURCE_KEY,
        "publisher_file": SOURCE_PDF,
        "publisher_file_sha256": SOURCE_SHA256,
        "publisher_pdf_url": (
            "https://www.sil.org/system/files/reapdata/14/18/32/"
            "141832844209183669770425266092693708252/silesr2019_006.pdf"
        ),
        "publisher_pdf_archive_capture": (
            "https://web.archive.org/web/20240616000000id_/https://www.sil.org/"
            "system/files/reapdata/14/18/32/"
            "141832844209183669770425266092693708252/silesr2019_006.pdf"
        ),
        "source_extent": "185 PDF pages; Appendix A.4 printed pp. 38-79 (PDF pp. 46-87)",
        "snapshot_file": str(SNAPSHOT.relative_to(REPO)),
        "snapshot_sha256": SNAPSHOT_SHA256,
        "source_prompts": 210,
        "source_response_records": len(source_rows),
        "source_lect_site_lists": 27,
        "target_lect_site_lists": 22,
        "target_response_records": 5056,
        "installed_responses": len(installed),
        "target_no_entry_records": 29,
        "audit_only_prior_or_control_records": 1150,
        "audit_records": len(audit),
        "audit_status_counts": dict(Counter(row["Status"] for row in audit)),
        "source_scope_counts": dict(Counter(row["Source_Scope"] for row in audit)),
        "installed_language_counts": dict(Counter(row["Language_ID"] for row in installed)),
        "unparsed_lines": 0,
        "wrapped_forms_joined_and_visually_checked": 10,
        "replacement_or_private_use_glyphs": 0,
        "ocr_used": False,
        "etymology_edges": 0,
        "coordinate_policy": (
            "modern GeoNames/OpenStreetMap or government locality points; "
            "the source supplies regional maps but no point coordinates; quality C"
        ),
    }
    return installed, audit, manifest


def write() -> None:
    installed, audit, manifest = build()
    with INSTALLED.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FORM_FIELDS)
        writer.writerows(installed)
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"installed={len(installed)} excluded={len(audit) - len(installed)} audit={len(audit)}")


if __name__ == "__main__":
    write()
