#!/usr/bin/env python3
"""Guard and stage the manually reviewed Noira 2015 source-local ledger.

This deliberately refuses OCR-bearing schemas and undeclared/unreviewed rows.
It does not write shared registries or generated CLDF files.
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
SOURCE_KEY = "varghesekumar2015noira"
PDF_SHA256 = "cb93db089a21e55e878f436632d8282c64c98fca85afe18179f8f3383db35280"
DECLARATION = "hand-keyed-from-rendered-source; OCR-not-copied"
ALLOWED_STATUSES = {"attested", "source_blank", "ambiguous", "illegible"}
CONTROL_CODES = {"GUJ", "MAR", "HIN"}
REPUBLISHED_CODES = {"NAS", "BMU", "NTO"}
LIST_REGISTRY = HERE / "list_registry.tsv"
PROFILE = HERE / "conversion_profile.tsv"
UNRESOLVED = HERE / "unresolved_readings.tsv"
STAGED_FORMS = HERE / "staged_forms.csv"
STAGED_AUDIT = HERE / "exhaustive_audit.tsv"
DUPLICATE_AUDIT = HERE / "dhule_republication_reconciliation.tsv"
MANIFEST = HERE / "source_manifest.json"
CHUNKS = HERE / "manual_chunks"
REQUIRED = {
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels", "Review_Status",
    "Confidence", "Uncertainty", "Reviewer_Method", "Reviewed_At",
    "Reviewer_Declaration",
}
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Item", "Gloss", "Site_Code", "Site_Name", "PDF_Page", "Printed_Page",
    "Column", "Manual_Transcription", "Source_Cognate_Labels", "Review_Status",
    "Confidence", "Uncertainty", "Reviewer_Method", "Reviewed_At",
    "Reviewer_Declaration", "Scope", "Disposition", "Citation",
    "Installed_Count", "Entry_Keys",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manual_cells(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        assert REQUIRED == fields, f"unexpected ledger schema: {sorted(fields)}"
        assert not any("ocr" in field.casefold() for field in fields)
        rows = list(reader)

    seen: set[tuple[str, str]] = set()
    for row in rows:
        assert all(unicodedata.is_normalized("NFC", value) for value in row.values())
        key = (row["Item"], row["Site_Code"])
        assert key not in seen, f"duplicate conceptual cell: {key}"
        seen.add(key)
        assert row["Reviewer_Declaration"] == DECLARATION
        assert row["Reviewer_Method"] == "manual visual inspection of 400-dpi rendered PDF page"
        assert row["Review_Status"] in ALLOWED_STATUSES
        assert row["Column"] in {"left", "right"}
        if row["Review_Status"] == "attested":
            assert row["Manual_Transcription"]
            assert row["Confidence"] in {"high", "medium"}
            assert len(row["Manual_Transcription"].split(" | ")) == len(
                row["Source_Cognate_Labels"].split(" | ")
            )
        else:
            assert not row["Manual_Transcription"]
            assert row["Uncertainty"]
    return rows


def load_all_manual_cells() -> list[dict[str, str]]:
    ledgers = sorted(CHUNKS.glob("items_*_hand_keyed.tsv"))
    rows = [row for ledger in ledgers for row in load_manual_cells(ledger)]
    expected = {(str(item), site) for item in range(1, 211) for site in [
        "NCH", "NPN", "NAS", "NGO", "BMU", "DBM", "DBA", "NTO", "KNA",
        "KTA", "GTA", "GUJ", "MAR", "HIN", "NTE", "TKO", "NJA",
    ]}
    keys = [(row["Item"], row["Site_Code"]) for row in rows]
    assert len(rows) == 3570, f"expected 3,570 reviewed cells, found {len(rows)}"
    assert len(keys) == len(set(keys)) and set(keys) == expected
    assert {int(row["Item"]) for row in rows} == set(range(1, 211))
    return rows


def load_registry() -> dict[str, dict[str, str]]:
    with LIST_REGISTRY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 17 and len({row["Site_Code"] for row in rows}) == 17
    assert Counter(row["Scope"] for row in rows) == Counter(
        new_target=11, republished_dhule=3, comparison_control=3
    )
    assert {row["Site_Code"] for row in rows if row["Scope"] == "republished_dhule"} == REPUBLISHED_CODES
    for row in rows:
        assert row["Install"] == ("yes" if row["Scope"] == "new_target" else "no")
        if row["Install"] == "yes":
            assert row["Language_ID"] and row["Dialect_ID"]
    return {row["Site_Code"]: row for row in rows}


def stage_target_forms(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    staged: list[dict[str, str]] = []
    for row in rows:
        if row["Site_Code"] in CONTROL_CODES or row["Review_Status"] != "attested":
            continue
        forms = row["Manual_Transcription"].split(" | ")
        labels = row["Source_Cognate_Labels"].split(" | ")
        for variant, (form, label) in enumerate(zip(forms, labels), 1):
            staged.append({
                "Item": row["Item"],
                "Gloss": row["Gloss"],
                "Site_Code": row["Site_Code"],
                "Form": form,
                "Source_Cognate_Label": label,
                "Variant": str(variant),
                "Source_Coordinates": (
                    f"PDF p.{row['PDF_Page']} / printed p.{row['Printed_Page']} / "
                    f"item {row['Item']} / {row['Site_Code']} / {row['Column']} column"
                ),
            })
    return staged


def dialect_tag(spec: dict[str, str]) -> str:
    return (
        f"dialect:{spec['Language_ID']}:"
        f"{quote(spec['Dialect_ID'], safe='')}:{quote(spec['Display_Name'], safe='')}"
    )


def build_package(
    rows: list[dict[str, str]], registry: dict[str, dict[str, str]]
) -> tuple[list[list[str]], list[dict[str, str]], dict[str, int]]:
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for row in rows:
        spec = registry[row["Site_Code"]]
        citation = (
            f"{SOURCE_KEY}[Appendix A3, printed p. {row['Printed_Page']}, "
            f"item {row['Item']}, list {row['Site_Code']}]"
        )
        entry_keys: list[str] = []
        if row["Review_Status"] == "source_blank":
            disposition = "blank-excluded"
        elif row["Review_Status"] in {"ambiguous", "illegible"}:
            disposition = "unresolved-excluded"
        elif spec["Scope"] == "comparison_control":
            disposition = "control-excluded"
        elif spec["Scope"] == "republished_dhule":
            disposition = "republished-dhule-excluded"
        else:
            disposition = "staged"
            source_forms = row["Manual_Transcription"].split(" | ")
            labels = row["Source_Cognate_Labels"].split(" | ")
            for variant, (form, label) in enumerate(zip(source_forms, labels), 1):
                entry_key = (
                    f"noira2015:p{int(row['PDF_Page']):03d}:"
                    f"i{int(row['Item']):03d}:{row['Site_Code']}:a{variant}"
                )
                entry_keys.append(entry_key)
                forms.append([
                    spec["Language_ID"], "", form, row["Gloss"], "", form,
                    f"source cognate/similarity group {label}; source list {row['Site_Name']}",
                    citation, "", "", entry_key, "", "", "",
                    dialect_tag(spec),
                ])
        audit.append({
            **row, "Scope": spec["Scope"], "Disposition": disposition,
            "Citation": citation, "Installed_Count": str(len(entry_keys)),
            "Entry_Keys": " | ".join(entry_keys),
        })

    counts = {
        "reviewed_cells": len(rows),
        "attested_cells": sum(row["Review_Status"] == "attested" for row in rows),
        "source_blank_cells": sum(row["Review_Status"] == "source_blank" for row in rows),
        "ambiguous_cells": sum(row["Review_Status"] == "ambiguous" for row in rows),
        "illegible_cells": sum(row["Review_Status"] == "illegible" for row in rows),
        "expanded_responses": sum(
            len(row["Manual_Transcription"].split(" | "))
            for row in rows if row["Review_Status"] == "attested"
        ),
        "new_target_conceptual_cells": sum(
            registry[row["Site_Code"]]["Scope"] == "new_target" for row in rows
        ),
        "new_target_attested_cells": sum(
            row["Review_Status"] == "attested"
            and registry[row["Site_Code"]]["Scope"] == "new_target"
            for row in rows
        ),
        "new_target_blank_cells": sum(
            row["Review_Status"] == "source_blank"
            and registry[row["Site_Code"]]["Scope"] == "new_target"
            for row in rows
        ),
        "installed_forms": len(forms),
        "republished_dhule_cells_excluded": sum(
            registry[row["Site_Code"]]["Scope"] == "republished_dhule" for row in rows
        ),
        "republished_dhule_responses_excluded": sum(
            len(row["Manual_Transcription"].split(" | "))
            for row in rows
            if row["Review_Status"] == "attested"
            and registry[row["Site_Code"]]["Scope"] == "republished_dhule"
        ),
        "control_cells_excluded": sum(
            registry[row["Site_Code"]]["Scope"] == "comparison_control" for row in rows
        ),
        "control_responses_excluded": sum(
            len(row["Manual_Transcription"].split(" | "))
            for row in rows
            if row["Review_Status"] == "attested"
            and registry[row["Site_Code"]]["Scope"] == "comparison_control"
        ),
    }
    assert counts == {
        "reviewed_cells": 3570, "attested_cells": 3526,
        "source_blank_cells": 44, "ambiguous_cells": 0, "illegible_cells": 0,
        "expanded_responses": 4385, "new_target_conceptual_cells": 2310,
        "new_target_attested_cells": 2271, "new_target_blank_cells": 39,
        "installed_forms": 2714, "republished_dhule_cells_excluded": 630,
        "republished_dhule_responses_excluded": 834,
        "control_cells_excluded": 630, "control_responses_excluded": 837,
    }
    return forms, audit, counts


def load_profile() -> list[tuple[str, str]]:
    with PROFILE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows and set(rows[0]) == {"Grapheme", "IPA"}
    graphemes = [row["Grapheme"] for row in rows]
    assert all(graphemes) and len(graphemes) == len(set(graphemes))
    return sorted(((row["Grapheme"], row["IPA"]) for row in rows), key=lambda x: -len(x[0]))


def validate_profile(forms: list[list[str]]) -> None:
    profile = load_profile()
    for row in forms:
        pending = row[FORM_FIELDS.index("Form")]
        while pending:
            match = next((grapheme for grapheme, _ in profile if pending.startswith(grapheme)), None)
            assert match is not None, f"unprofiled sequence in {pending!r} from {row!r}"
            pending = pending[len(match):]


def build_republication_audit(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    dhule_path = HERE.parent / "sil_northern_dhule_bhils_2013" / "staged_audit.tsv"
    assert dhule_path.exists(), "Northern Dhule source-local audit is required for reconciliation"
    with dhule_path.open(encoding="utf-8", newline="") as handle:
        dhule = list(csv.DictReader(handle, delimiter="\t"))
    mapping = {"NAS": "AST", "BMU": "MUN", "NTO": "TOR"}
    dhule_by_key = {(row["Item"], row["Site_Code"]): row for row in dhule}
    reconciled = []
    for row in rows:
        if row["Site_Code"] not in mapping:
            continue
        older = dhule_by_key[(row["Item"], mapping[row["Site_Code"]])]
        same = row["Manual_Transcription"] == older["Manual_Transcription"]
        reconciled.append({
            "Item": row["Item"], "Gloss": row["Gloss"],
            "Noira_Site": row["Site_Code"], "Dhule_Site": mapping[row["Site_Code"]],
            "Noira_PDF_Page": row["PDF_Page"], "Dhule_PDF_Page": older["PDF_Page"],
            "Noira_Manual_Transcription": row["Manual_Transcription"],
            "Dhule_Manual_Transcription": older["Manual_Transcription"],
            "Comparison": "literal-ledger-exact" if same else "same-source-representation-differs",
            "Disposition": "exclude Noira republication; retain primary ESR 2013-004 route",
        })
    assert len(reconciled) == 630
    return reconciled


def write_package(
    forms: list[list[str]], audit: list[dict[str, str]],
    reconciliation: list[dict[str, str]], counts: dict[str, int],
) -> None:
    with STAGED_FORMS.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(forms)
    with STAGED_AUDIT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(audit)
    with DUPLICATE_AUDIT.open("w", encoding="utf-8", newline="") as handle:
        fields = list(reconciliation[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(reconciliation)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["lexical_appendix"].update({
        "regional_lists": 14,
        "new_target_lists": 11,
        "new_target_cells": 2310,
        "republished_dhule_lists": 3,
        "republished_dhule_cells": 630,
        "control_lists": 3,
        "control_cells": 630,
    })
    manifest["lexical_appendix"].pop("target_lists", None)
    manifest["lexical_appendix"].pop("target_cells", None)
    manifest["manual_review"] = {
        "completed_items": "1-210", "remaining_cells": 0,
        **counts,
        "method": (
            "manual visual inspection of 400-dpi rendered PDF pages; selected difficult "
            "glyphs rechecked at 800 or 900 dpi; OCR/PDF text not accepted"
        ),
        "unresolved": [],
    }
    manifest["artifacts"] = {
        "staged_forms": {"path": STAGED_FORMS.name, "sha256": sha256(STAGED_FORMS)},
        "exhaustive_audit": {"path": STAGED_AUDIT.name, "sha256": sha256(STAGED_AUDIT)},
        "republication_reconciliation": {
            "path": DUPLICATE_AUDIT.name, "sha256": sha256(DUPLICATE_AUDIT),
        },
        "list_registry": {"path": LIST_REGISTRY.name, "sha256": sha256(LIST_REGISTRY)},
        "conversion_profile": {"path": PROFILE.name, "sha256": sha256(PROFILE)},
        "unresolved_readings": {"path": UNRESOLVED.name, "sha256": sha256(UNRESOLVED)},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger", type=Path,
        default=CHUNKS / "items_001_027_hand_keyed.tsv",
    )
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--all", action="store_true", help="validate the complete 210 x 17 package")
    parser.add_argument("--write", action="store_true", help="write source-local staging and audit artifacts")
    args = parser.parse_args()
    if args.pdf:
        assert sha256(args.pdf) == PDF_SHA256, "canonical PDF checksum mismatch"
    rows = load_all_manual_cells() if args.all or args.write else load_manual_cells(args.ledger)
    if args.all or args.write:
        registry = load_registry()
        forms, audit, counts = build_package(rows, registry)
        validate_profile(forms)
        reconciliation = build_republication_audit(rows)
        if args.write:
            write_package(forms, audit, reconciliation, counts)
        print(" ".join(f"{key}={value}" for key, value in counts.items()))
        return
    staged = stage_target_forms(rows)
    print(
        f"guarded ledger OK: {len(rows)} reviewed cells; "
        f"{sum(r['Review_Status'] == 'attested' for r in rows)} attested; "
        f"{sum(r['Review_Status'] == 'source_blank' for r in rows)} source blanks; "
        f"{len(staged)} target form candidates"
    )


if __name__ == "__main__":
    main()
