#!/usr/bin/env python3
"""Install only target-scope attestations from the frozen Kurumba package.

The 4,738-row ``staged_forms.csv`` and 10,450-row ``staged_audit.csv`` are
immutable inputs.  This integration filter never reads OCR fields or changes a
manual form; it excludes every comparison-control attestation, printed dash,
ambiguous cell, and illegible cell with an explicit cell-addressed disposition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import unicodedata
from collections import Counter
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE.parents[4]
FROZEN_FORMS = PACKAGE / "staged_forms.csv"
FROZEN_AUDIT = PACKAGE / "staged_audit.csv"
TARGET_FORMS = PACKAGE / "installed_target_forms.csv"
INTEGRATION_AUDIT = PACKAGE / "shared_integration_audit.csv"
INTEGRATION_MANIFEST = PACKAGE / "shared_integration_manifest.json"
SHARED_FORMS = DATA_ROOT / "data/other/forms/20260828-sil-kurumba.csv"
SHARED_PROFILE = DATA_ROOT / "conversion/sil-kurumba-2012.txt"
SOUND_INVENTORY = PACKAGE / "sound_inventory.tsv"
SOUND_DECISIONS = PACKAGE / "sound_profile_decisions.json"

FROZEN_FORMS_SHA256 = "71e985690e76497ae276d43ad93b7a44ab75565ba9bd88f11de6a5d04a43b29b"
FROZEN_AUDIT_SHA256 = "3ed3963bdbc309ec3926589e70942c9b52acb8a17b8687d2e47a503ea2e743bf"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_forms() -> list[list[str]]:
    with FROZEN_FORMS.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def read_audit() -> list[dict[str, str]]:
    with FROZEN_AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def classify(row: dict[str, str]) -> tuple[str, str]:
    scope, cell_status, staging_status = row["Scope"], row["Cell_Status"], row["Status"]
    if staging_status == "staged" and cell_status == "attested":
        if scope == "target":
            return "installed_target", "audited attestation from a target list"
        if scope == "control":
            return "excluded_control", "comparison-control attestation retained audit-only"
    if staging_status == "missing" and cell_status == "blank":
        return (
            ("excluded_blank_target", "printed dash in a target list")
            if scope == "target"
            else ("excluded_blank_control", "printed dash in a comparison-control list")
        )
    if staging_status == "unresolved" and scope == "target" and cell_status in {"ambiguous", "illegible"}:
        return f"excluded_{cell_status}_target", f"unresolved {cell_status} target cell; no form inferred"
    raise AssertionError((scope, cell_status, staging_status, row["Cell_Key"]))


def build() -> tuple[list[list[str]], list[dict[str, str]], Counter[str]]:
    assert sha256(FROZEN_FORMS) == FROZEN_FORMS_SHA256
    assert sha256(FROZEN_AUDIT) == FROZEN_AUDIT_SHA256
    forms = read_forms()
    audit = read_audit()
    assert len(forms) == 4738 and all(len(row) == 15 for row in forms)
    assert len(audit) == 10450
    by_key = {row[10]: row for row in forms}
    assert len(by_key) == 4738
    staged_audit = {row["Entry_Key"]: row for row in audit if row["Status"] == "staged"}
    assert by_key.keys() == staged_audit.keys()
    for key, form in by_key.items():
        evidence = staged_audit[key]
        assert form[0] == evidence["Language_ID"]
        assert form[2] == form[5] == evidence["Manual_Form"]
        assert form[3] == evidence["Manual_Gloss"]
        assert form[7] == evidence["Citation"]
        assert form[10] == evidence["Entry_Key"]
        assert evidence["Dialect_ID"] in form[14]

    dispositions: Counter[str] = Counter()
    integrated_audit: list[dict[str, str]] = []
    target_keys: set[str] = set()
    for row in audit:
        disposition, reason = classify(row)
        dispositions[disposition] += 1
        if disposition == "installed_target":
            target_keys.add(row["Entry_Key"])
        integrated_audit.append({
            **row,
            "Integration_Status": disposition,
            "Integration_Reason": reason,
        })

    assert dispositions == {
        "installed_target": 3204,
        "excluded_control": 1534,
        "excluded_blank_target": 5044,
        "excluded_blank_control": 666,
        "excluded_ambiguous_target": 1,
        "excluded_illegible_target": 1,
    }
    target_forms = [row for row in forms if row[10] in target_keys]
    assert len(target_forms) == len(target_keys) == 3204
    assert all(staged_audit[row[10]]["Scope"] == "target" for row in target_forms)
    return target_forms, integrated_audit, dispositions


def write_sound_evidence(frozen_forms: list[list[str]], target_forms: list[list[str]]) -> None:
    frozen_counts = Counter(char for row in frozen_forms for char in row[2])
    target_counts = Counter(char for row in target_forms for char in row[2])
    with SOUND_INVENTORY.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "Character", "Codepoint", "Unicode_Name", "Frozen_Attested_Count",
            "Installed_Target_Count", "Combining_Class",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for char in sorted(frozen_counts, key=ord):
            writer.writerow({
                "Character": char,
                "Codepoint": f"U+{ord(char):04X}",
                "Unicode_Name": unicodedata.name(char, "UNKNOWN"),
                "Frozen_Attested_Count": frozen_counts[char],
                "Installed_Target_Count": target_counts[char],
                "Combining_Class": unicodedata.combining(char),
            })
    decisions = {
        "input_layer": "manual source transcription from rendered Appendix C cells",
        "output_layer": "Jambu display transcription; source form remains unchanged in Phonemic",
        "base_profile": "conversion/sil-survey.txt",
        "base_profile_sha256": "26526fb668d42e016e602372396b68c0f77ac5d665d01f0f55bfd198f4874677",
        "installed_profile": "conversion/sil-kurumba-2012.txt",
        "installed_profile_sha256": sha256(SHARED_PROFILE),
        "inventory": {
            "frozen_attested_forms": len(frozen_forms),
            "installed_target_forms": len(target_forms),
            "unique_codepoints": len(frozen_counts),
        },
        "decisions": [
            {
                "inputs": ["a:", "ʌ:", "i:", "ɪ:", "u:", "e:", "ɛ:", "o:", "ɔ:", "ə:", "ɨ:"],
                "outputs": ["ā", "ā", "ī", "ī", "ū", "ē", "ē", "ō", "ō", "ə̄", "ɨ̄"],
                "reason": "The printed ASCII colon is the source length mark when it immediately follows a vowel.",
            },
            {
                "inputs": ["'", ":"],
                "outputs": ["'", ":"],
                "reason": "Preserve the raised apostrophe and the single residual post-apostrophe colon because the scan does not establish a safer phonetic interpretation.",
            },
            {
                "inputs": ["c", "z", "y", "ɣ", "β", "ṇ", "ṭ", "-"],
                "outputs": ["c", "z", "y", "ɣ", "β", "ṇ", "ṭ", "-"],
                "reason": "Identity-preserve visibly distinct source symbols and punctuation not normalized by the survey base profile.",
            },
            {
                "inputs": ["ɽ"],
                "outputs": ["ṛ"],
                "reason": "Apply the established Jambu house symbol for the retroflex flap.",
            },
        ],
        "unresolved_profile_mappings": [],
    }
    SOUND_DECISIONS.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write(target_forms: list[list[str]], audit: list[dict[str, str]], dispositions: Counter[str]) -> None:
    with TARGET_FORMS.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(target_forms)
    fields = list(audit[0])
    with INTEGRATION_AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    frozen_forms = read_forms()
    write_sound_evidence(frozen_forms, target_forms)
    manifest = {
        "state": "shared_source_specific_integration_complete",
        "policy": "Frozen manual readings are authoritative; install target scope only; controls and unresolved cells remain audit-only.",
        "source_key": "blairetal2012kurumba",
        "source_pdf_sha256": "250dc3d83661227caa66bf16e390e51c2dcb7186fa435252541ed13bbfcd9137",
        "frozen_forms": {"rows": 4738, "sha256": FROZEN_FORMS_SHA256},
        "frozen_audit": {"rows": 10450, "sha256": FROZEN_AUDIT_SHA256},
        "installed_target_forms": {"rows": 3204, "sha256": sha256(TARGET_FORMS)},
        "integration_audit": {"rows": 10450, "sha256": sha256(INTEGRATION_AUDIT)},
        "sound_profile": {
            "path": "conversion/sil-kurumba-2012.txt",
            "sha256": sha256(SHARED_PROFILE),
            "inventory_sha256": sha256(SOUND_INVENTORY),
            "decisions_sha256": sha256(SOUND_DECISIONS),
            "unresolved_mappings": [],
        },
        "dispositions": dict(sorted(dispositions.items())),
        "unresolved_coordinates": [
            "kurumba2012:pudukkottai:i020",
            "kurumba2012:kotagiri_alu:i025",
        ],
        "post_freeze_reconciliation": {
            "prior_rows_with_source_key": 0,
            "prior_rows_with_kurumba2012_entry_key": 0,
            "result": "new source installation; no pre-existing row competed with the frozen target output",
        },
        "deferred_gates": [
            "consolidated CLDF/full build",
            "opaque f_* identity reconciliation",
            "repository-wide graph validation and full test suite",
            "browser database refresh and QA",
            "commit and shipping",
        ],
    }
    INTEGRATION_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-shared", action="store_true")
    args = parser.parse_args()
    target_forms, audit, dispositions = build()
    write(target_forms, audit, dispositions)
    if args.install_shared:
        shutil.copyfile(TARGET_FORMS, SHARED_FORMS)
        assert sha256(SHARED_FORMS) == sha256(TARGET_FORMS)
    print(
        f"target_forms={len(target_forms)} audit_rows={len(audit)} "
        f"target_sha256={sha256(TARGET_FORMS)}"
    )


if __name__ == "__main__":
    main()
