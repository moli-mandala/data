#!/usr/bin/env python3
"""Reconcile the frozen manual ledger and prepare source-local staging artifacts.

Legacy data are comparison-only inputs. They never supply or validate a manual form.
"""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE.parents[4]
MANUAL_MANIFEST = PACKAGE / "source_manifest.json"
LEGACY_INSTALLED = PACKAGE / "legacy_20260826-sil-kurux.csv"
LEGACY_AUDIT = PACKAGE / "legacy_20260826-sil-kurux-audit.csv"
BASE_PROFILE = PACKAGE / "legacy_sound_profile_base.txt"
RECONCILIATION = PACKAGE / "reconciliation.tsv"
STAGING_AUDIT = PACKAGE / "staging_audit.tsv"
STAGED_FORMS = PACKAGE / "staged_forms.csv"
SITE_METADATA = PACKAGE / "site_metadata.tsv"
REFERENCE_METADATA = PACKAGE / "reference_metadata.json"
EXCLUSION_POLICY = PACKAGE / "exclusion_policy.json"
SOUND_INVENTORY = PACKAGE / "sound_inventory.tsv"
SOUND_PROFILE = PACKAGE / "sound_profile.txt"
SOUND_DECISIONS = PACKAGE / "sound_profile_decisions.json"
POST_FREEZE_MANIFEST = PACKAGE / "post_freeze_manifest.json"

SOURCE_KEY = "kim-ahmad-kim-sangma2011kurux"
SOURCE_PDF_SHA256 = "f2f06c25ac55462d6a40843539d8417e24a647bd1eb0bbe3f24ea3e45f0b9e4b"
SITE_ORDER = {code: index for index, code in enumerate("ABCDE0")}
SITES = {
    "A": {
        "site_id": "kurux2011-A-dima",
        "site_name": "Dima",
        "role": "target",
        "language_id": "Kurux",
        "glottocode": "kuru1301",
        "dialect_tag": "dialect:Kurux:kurux2011-A-dima:Dima%20%28Dima%2C%20West%20Bengal%2C%20India%29",
        "location": "Dima, West Bengal, India",
        "administrative_context": "The report identifies Dima only as a West Bengal site.",
        "evidence": "physical p. 13 / printed p. 12, Table 2",
    },
    "B": {
        "site_id": "kurux2011-B-gabindanagar",
        "site_name": "Gabindanagar",
        "role": "target",
        "language_id": "Kurux",
        "glottocode": "kuru1301",
        "dialect_tag": "dialect:Kurux:kurux2011-B-gabindanagar:Gabindanagar%20%28Gabindanagar%2C%20Bangladesh%29",
        "location": "Gabindanagar, Thakurgaon sub-district, Thakurgaon district, Bangladesh",
        "administrative_context": "One of two visited villages in the Thakurgaon survey area.",
        "evidence": "physical p. 13 / printed p. 12, Table 2; physical p. 15 / printed p. 14",
    },
    "C": {
        "site_id": "kurux2011-C-boldipukur",
        "site_name": "Boldipukur",
        "role": "target",
        "language_id": "Kurux",
        "glottocode": "kuru1301",
        "dialect_tag": "dialect:Kurux:kurux2011-C-boldipukur:Boldipukur%20%28Boldipukur%2C%20Bangladesh%29",
        "location": "Boldipukur/Tajnagar, Mithapukur sub-district, near Rangpur, Bangladesh",
        "administrative_context": "The report describes the village area as near a Catholic mission and quite close to Rangpur city.",
        "evidence": "physical p. 13 / printed p. 12, Table 2; physical p. 15 / printed p. 14",
    },
    "D": {
        "site_id": "kurux2011-D-lohanipara",
        "site_name": "Lohanipara",
        "role": "target",
        "language_id": "Kurux",
        "glottocode": "kuru1301",
        "dialect_tag": "dialect:Kurux:kurux2011-D-lohanipara:Lohanipara%20%28Lohanipara%2C%20Bangladesh%29",
        "location": "Lohanipara, Bodorganj sub-district, Rangpur area, Bangladesh",
        "administrative_context": "The report describes it as the relatively remote westernmost part of the Rangpur survey area.",
        "evidence": "physical p. 13 / printed p. 12, Table 2; physical p. 15 / printed p. 14",
    },
    "E": {
        "site_id": "kurux2011-E-dulhapur",
        "site_name": "Dulhapur",
        "role": "target",
        "language_id": "Kurux",
        "glottocode": "kuru1301",
        "dialect_tag": "dialect:Kurux:kurux2011-E-dulhapur:Dulhapur%20%28Dulhapur%2C%20Bangladesh%29",
        "location": "Dulhapur/Rameswarpara, Mithapukur sub-district, Bangladesh",
        "administrative_context": "The report describes it as the southernmost Kurux village area in the survey region.",
        "evidence": "physical p. 13 / printed p. 12, Table 2; physical p. 15 / printed p. 14",
    },
    "0": {
        "site_id": "kurux2011-0-bangla",
        "site_name": "Bangla",
        "role": "control",
        "language_id": "B",
        "glottocode": "beng1280",
        "dialect_tag": "dialect:B:kurux2011-0-bangla:Bangla%20%28Bangla%29",
        "location": "Standard Bangla comparison list; no elicitation locality claimed",
        "administrative_context": "The report explicitly calls code 0 the standard Bangla wordlist.",
        "evidence": "physical p. 13 / printed p. 12, Table 2",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_manual() -> tuple[dict, list[dict[str, str]]]:
    manifest = json.loads(MANUAL_MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for chunk in manifest["chunks"]:
        rows.extend(read_tsv(PACKAGE / chunk["expanded_cells"]))
    assert manifest["state"] == "manual_review_complete"
    assert manifest["pending_items"] == []
    assert len(rows) == 1869
    assert len({(row["item"], row["site_code"]) for row in rows}) == 1842
    return manifest, rows


def load_legacy() -> tuple[list[dict[str, str]], list[list[str]]]:
    with LEGACY_AUDIT.open(encoding="utf-8", newline="") as stream:
        audit = list(csv.DictReader(stream))
    with LEGACY_INSTALLED.open(encoding="utf-8", newline="") as stream:
        installed = list(csv.reader(stream))
    assert len(audit) == 1809
    assert len(installed) == 1422
    assert all(len(row) == 15 for row in installed)
    return audit, installed


def expand_legacy(audit: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    ranks: Counter[tuple[int, str]] = Counter()
    for raw in audit:
        if raw["Site_Code"]:
            codes = raw["Site_Code"]
        else:
            assert raw["Reason"] == "printed gap: the item was not elicited at any site"
            codes = "ABCDE0"
        for code in codes:
            coordinate = (int(raw["Item"]), code)
            ranks[coordinate] += 1
            if raw["Status"] == "installed":
                status = "attested"
            elif raw["Reason"] == "contains a glyph with no verified reading":
                status = "unresolved"
            elif raw["Reason"] == "printed gap: the item was not elicited at this site":
                status = "blank"
            else:
                status = "not_used"
            rows.append({
                **raw,
                "expanded_site_code": code,
                "expanded_variant": str(ranks[coordinate]),
                "normalized_status": status,
                "normalized_group": raw["Group"] or "0",
            })
    assert len(rows) == 1869
    return rows


def build_reconciliation(
    manual: list[dict[str, str]], legacy: list[dict[str, str]]
) -> tuple[list[dict[str, str]], Counter[str]]:
    manual_by_key = {
        (int(row["item"]), row["site_code"], int(row["site_variant"])): row
        for row in manual
    }
    legacy_by_key = {
        (int(row["Item"]), row["expanded_site_code"], int(row["expanded_variant"])): row
        for row in legacy
    }
    assert manual_by_key.keys() == legacy_by_key.keys()
    rows = []
    counts: Counter[str] = Counter()
    for key in sorted(manual_by_key, key=lambda k: (k[0], SITE_ORDER[k[1]], k[2])):
        current = manual_by_key[key]
        old = legacy_by_key[key]
        if current["status"] == "attested" and old["normalized_status"] == "unresolved":
            comparison = "manual_recovered_legacy_unresolved"
        elif current["status"] == "attested" and old["normalized_status"] == "attested":
            comparison = "form_exact" if current["form"] == old["Raw_Form"] else "form_difference"
        elif current["status"] == old["normalized_status"] == "blank":
            comparison = "blank_match"
        elif current["status"] == old["normalized_status"] == "not_used":
            comparison = "not_used_match"
        else:
            raise AssertionError((key, current, old))
        counts[comparison] += 1
        entry_key = f"silkurux2011:i{key[0]:03d}:{key[1]}:{key[2]}"
        rows.append({
            "item": str(key[0]),
            "site_code": key[1],
            "site_variant": str(key[2]),
            "manual_entry_key": entry_key,
            "manual_physical_page": current["physical_page"],
            "manual_printed_page": current["printed_page"],
            "manual_line_id": current["line_id"],
            "manual_gloss": current["gloss"],
            "manual_group": current["group"] or "0",
            "manual_status": current["status"],
            "manual_form": current["form"],
            "legacy_pdf_page": old["PDF_Page"],
            "legacy_group": old["normalized_group"],
            "legacy_status": old["normalized_status"],
            "legacy_raw_form": old["Raw_Form"],
            "legacy_entry_key": old["Entry_Key"],
            "legacy_reason": old["Reason"],
            "gloss_match": str(current["gloss"] == old["Gloss"]).lower(),
            "group_match": str((current["group"] or "0") == old["normalized_group"]).lower(),
            "comparison": comparison,
        })
    assert counts == {
        "form_exact": 722,
        "form_difference": 700,
        "manual_recovered_legacy_unresolved": 239,
        "blank_match": 136,
        "not_used_match": 72,
    }
    assert all(row["gloss_match"] == "true" for row in rows)
    assert all(row["group_match"] == "true" for row in rows)
    return rows, counts


def build_staging(manual: list[dict[str, str]]) -> tuple[list[list[str]], list[dict[str, str]], Counter[str]]:
    staged = []
    audit = []
    counts: Counter[str] = Counter()
    for row in sorted(manual, key=lambda r: (
        int(r["item"]), SITE_ORDER[r["site_code"]], int(r["site_variant"])
    )):
        code = row["site_code"]
        site = SITES[code]
        key = f"silkurux2011:i{int(row['item']):03d}:{code}:{row['site_variant']}"
        if row["status"] == "blank":
            disposition = "excluded_blank"
        elif row["status"] == "not_used":
            disposition = "excluded_not_used"
        elif site["role"] == "control":
            disposition = "excluded_control"
        else:
            disposition = "staged_target"
            citation = (
                f"{SOURCE_KEY}[p. {row['printed_page']}, wordlist item {row['item']}, "
                f"site {code} {site['site_name']}]"
            )
            staged.append([
                "Kurux", "", row["form"], row["gloss"], "", row["form"],
                f"lexical-similarity group {row['group']}", citation, "", "", key,
                "", "", "", site["dialect_tag"],
            ])
        counts[disposition] += 1
        audit.append({
            "entry_key": key,
            "item": row["item"],
            "site_code": code,
            "site_variant": row["site_variant"],
            "role": site["role"],
            "language_id": site["language_id"],
            "dialect_id": site["site_id"],
            "status": row["status"],
            "form": row["form"],
            "gloss": row["gloss"],
            "physical_page": row["physical_page"],
            "printed_page": row["printed_page"],
            "line_id": row["line_id"],
            "disposition": disposition,
            "reason": {
                "staged_target": "attested Kurux target-site form",
                "excluded_blank": "printed no-entry cell",
                "excluded_not_used": "item printed [not used] for all sites",
                "excluded_control": "standard Bangla comparison control retained audit-only",
            }[disposition],
        })
    assert counts == {
        "staged_target": 1365,
        "excluded_control": 296,
        "excluded_blank": 136,
        "excluded_not_used": 72,
    }
    assert len(staged) == 1365 and all(len(row) == 15 for row in staged)
    return staged, audit, counts


def write_sites() -> None:
    fields = [
        "site_code", "site_id", "site_name", "role", "language_id", "glottocode",
        "dialect_tag", "location", "administrative_context", "latitude", "longitude",
        "coordinate_quality", "coordinate_note", "evidence",
    ]
    rows = []
    for code in "ABCDE0":
        site = SITES[code]
        rows.append({
            "site_code": code,
            **site,
            "latitude": "",
            "longitude": "",
            "coordinate_quality": "",
            "coordinate_note": "The report supplies map context but no exact site coordinate; no coordinate is invented.",
        })
    write_tsv(SITE_METADATA, rows, fields)


def write_reference() -> None:
    metadata = {
        "id": SOURCE_KEY,
        "entry_type": "techreport",
        "title": "The Kurux of Bangladesh: A Sociolinguistic Survey",
        "authors": ["Amy Kim", "Mridul Ahmad", "Seung Kim", "Palash Roy Sangma"],
        "institution": "SIL International",
        "series": "SIL Electronic Survey Report",
        "number": "2011-040",
        "year": 2011,
        "official_archive_url": "https://www.silbangladesh.org/resources/archives/41654",
        "source_pdf_sha256": SOURCE_PDF_SHA256,
        "source_pdf_pages": 90,
        "included": "Appendix A.3 items 1-307 at Kurux target sites A-E.",
        "excluded": (
            "Standard Bangla site 0 is retained audit-only; printed no-entry cells and globally "
            "unused items are accounted for but not staged; Appendix B onward is outside scope."
        ),
        "license": "Freely published by SIL International; only extracted linguistic facts are staged.",
        "ocr": "No. Every lexical reading was transcribed manually from rendered pages; OCR and PDF text were locator-only.",
        "provenance": [
            "data/other/forms/raw_data/sil_kurux_2011_manual/source_manifest.json",
            "data/other/forms/raw_data/sil_kurux_2011_manual/reconciliation.tsv",
            "data/other/forms/raw_data/sil_kurux_2011_manual/staging_audit.tsv",
            "data/other/forms/raw_data/sil_kurux_2011_manual/staged_forms.csv",
        ],
        "etymology_provenance": "none; the report presents lexical-similarity groups, not etymological claims",
        "jambu_editor": "Aryaman Arora and OpenAI Codex",
    }
    REFERENCE_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_exclusions(counts: Counter[str]) -> None:
    policy = {
        "source_rows": 1869,
        "staged_rows": counts["staged_target"],
        "dispositions": dict(sorted(counts.items())),
        "control_policy": "Code 0 is the report's standard Bangla comparison list and is audit-only.",
        "blank_policy": "Printed no-entry cells remain explicit audit rows and are not staged.",
        "not_used_policy": "Every globally unused item remains six explicit audit coordinates and is not staged.",
        "ambiguity_policy": "No ambiguous or illegible lexical coordinate remains after manual review.",
        "etymology_policy": "No cognate or borrowing edge is inferred from lexical-similarity groups.",
    }
    EXCLUSION_POLICY.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_sound_artifacts(manual: list[dict[str, str]], staged: list[list[str]]) -> None:
    all_forms = [row["form"] for row in manual if row["status"] == "attested"]
    staged_forms = [row[2] for row in staged]
    all_counts = Counter(char for form in all_forms for char in form)
    staged_counts = Counter(char for form in staged_forms for char in form)
    inventory = []
    for char in sorted(all_counts, key=ord):
        inventory.append({
            "character": char,
            "codepoint": f"U+{ord(char):04X}",
            "unicode_name": unicodedata.name(char, "UNKNOWN"),
            "all_attested_count": str(all_counts[char]),
            "staged_target_count": str(staged_counts[char]),
            "combining_class": str(unicodedata.combining(char)),
        })
    write_tsv(SOUND_INVENTORY, inventory, [
        "character", "codepoint", "unicode_name", "all_attested_count",
        "staged_target_count", "combining_class",
    ])

    base_lines = BASE_PROFILE.read_text(encoding="utf-8").splitlines()
    assert base_lines[0] == "Grapheme\tIPA"
    additions = [
        ("t͜ʃ", "c", "manual source uses a tie bar below"),
        ("d͜ʒ", "j", "manual source uses a tie bar below"),
        ("͜", "", "consume a residual tie bar only after multigraph rules"),
        ("ⁱ", "ⁱ", "preserve printed superscript vowel"),
        ("ᵘ", "ᵘ", "preserve printed superscript vowel"),
        ("ü", "ü", "preserve printed front rounded vowel"),
        ("ĩ", "ĩ", "preserve printed precomposed nasal vowel"),
        ("ʼ", "ʼ", "preserve printed modifier apostrophe"),
    ]
    profile = [base_lines[0]] + [f"{a}\t{b}" for a, b, _ in additions] + base_lines[1:]
    SOUND_PROFILE.write_text("\n".join(profile) + "\n", encoding="utf-8")
    decisions = {
        "input_layer": "manually transcribed source IPA",
        "output_layer": "Jambu display transcription",
        "phonemic_policy": "Preserve the manual source IPA in Phonemic and raw Form/Original; convert only display Form.",
        "base_profile": "conversion/sil-kurux.txt",
        "base_profile_sha256": sha256(BASE_PROFILE),
        "source_local_profile": "sound_profile.txt",
        "additions": [
            {"grapheme": a, "output": b, "reason": reason} for a, b, reason in additions
        ],
        "inventory_scope": {
            "all_attested_rows": len(all_forms),
            "staged_target_rows": len(staged_forms),
            "unique_codepoints": len(inventory),
        },
        "unresolved_mappings": [],
    }
    SOUND_DECISIONS.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    manual_manifest, manual = load_manual()
    legacy_audit, legacy_installed = load_legacy()
    legacy = expand_legacy(legacy_audit)
    reconciliation, reconciliation_counts = build_reconciliation(manual, legacy)
    write_tsv(RECONCILIATION, reconciliation, list(reconciliation[0]))
    staged, staging_audit, staging_counts = build_staging(manual)
    with STAGED_FORMS.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(staged)
    write_tsv(STAGING_AUDIT, staging_audit, list(staging_audit[0]))
    write_sites()
    write_reference()
    write_exclusions(staging_counts)
    write_sound_artifacts(manual, staged)

    legacy_keys = {row[10] for row in legacy_installed}
    audit_keys = {row["Entry_Key"] for row in legacy_audit if row["Status"] == "installed"}
    assert legacy_keys == audit_keys and len(legacy_keys) == 1422
    outputs = [
        RECONCILIATION, STAGING_AUDIT, STAGED_FORMS, SITE_METADATA,
        REFERENCE_METADATA, EXCLUSION_POLICY, SOUND_INVENTORY,
        SOUND_PROFILE, SOUND_DECISIONS,
    ]
    manifest = {
        "state": "source_local_post_freeze_complete",
        "policy": "Manual readings are authoritative; legacy and PDF text are comparison/locator-only.",
        "manual_manifest_sha256": sha256(MANUAL_MANIFEST),
        "legacy_installed_sha256": sha256(LEGACY_INSTALLED),
        "legacy_audit_sha256": sha256(LEGACY_AUDIT),
        "manual_conceptual_cells": manual_manifest["conceptual_cells"],
        "manual_expanded_rows": len(manual),
        "reconciliation_counts": dict(sorted(reconciliation_counts.items())),
        "staging_counts": dict(sorted(staging_counts.items())),
        "legacy_installed_rows": len(legacy_installed),
        "legacy_audit_rows": len(legacy_audit),
        "site_identity_state": "resolved_from_report_table_2",
        "coordinate_state": "exact_coordinates_not_printed; source-local metadata leaves them blank",
        "unresolved_lexical_coordinates": [],
        "deferred_shared_actions": [
            "replace legacy installed source CSV",
            "update shared dialect registry descriptions/coordinates",
            "update shared BibTeX entry",
            "route the source-local sound profile in the shared build",
            "run consolidated build and graph validation",
            "run browser database build and QA",
        ],
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    POST_FREEZE_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("reconciled 1869 rows; staged 1365 Kurux target attestations")


if __name__ == "__main__":
    main()
