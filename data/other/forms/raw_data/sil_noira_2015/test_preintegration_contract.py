from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import preintegration_audit as pre  # noqa: E402


def read_dicts(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def test_preintegration_artifacts_are_fresh_and_pdf_is_pinned() -> None:
    result = subprocess.run(
        [sys.executable, str(HERE / "preintegration_audit.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "cells=3570" in result.stdout
    assert "staged_forms=2714" in result.stdout
    assert "unresolved=0" in result.stdout
    pdf = pre.pdf_metadata()
    assert pdf == {
        "path": "tmp/pdfs/noira_2015/silesr2015_012.pdf",
        "bytes": 1676716,
        "pages": 96,
        "sha256": pre.PDF_HASH,
        "title": pre.PDF_TITLE,
        "author": pre.PDF_AUTHOR,
        "page_size_points": "612 x 792",
    }


def test_frozen_hashes_and_manual_census() -> None:
    rows, audit, forms = pre.verify_frozen_package()
    assert len(rows) == len(audit) == 3570
    assert Counter(row["Review_Status"] for row in rows) == Counter(
        attested=3526, source_blank=44
    )
    assert len(forms) == len({row[10] for row in forms}) == 2714
    for relative, expected in pre.FROZEN_HASHES.items():
        assert pre.sha256(HERE / relative) == expected
    assert not read_dicts(HERE / "unresolved_readings.tsv", "\t")


def test_scope_dispositions_and_expansion_are_exhaustive() -> None:
    audit = read_dicts(HERE / "exhaustive_audit.tsv", "\t")
    assert Counter(row["Scope"] for row in audit) == Counter(
        new_target=2310, republished_dhule=630, comparison_control=630
    )
    assert Counter(row["Disposition"] for row in audit) == Counter(
        staged=2271,
        **{
            "blank-excluded": 44,
            "republished-dhule-excluded": 625,
            "control-excluded": 630,
        },
    )
    assert sum(int(row["Installed_Count"]) for row in audit) == 2714
    assert all(
        row["Installed_Count"] == "0" and not row["Entry_Keys"]
        for row in audit if row["Scope"] != "new_target"
    )
    blanks = [row for row in audit if row["Review_Status"] == "source_blank"]
    assert len(blanks) == 44
    assert Counter(row["Scope"] for row in blanks) == Counter(new_target=39, republished_dhule=5)


def test_kotli_lists_use_provisional_noiri_dialect_routing() -> None:
    registry = {row["Site_Code"]: row for row in read_dicts(HERE / "list_registry.tsv", "\t")}
    for site in ["KNA", "KTA"]:
        assert registry[site]["Language_ID"] == "Noiri"
        assert "Provisional source-supported Noiri dialect routing" in registry[site]["Note"]
        assert "historical Kotali/Khandesi" in registry[site]["Note"]
    assert "Adivasi Bhil-Taradi" in registry["KTA"]["Note"]
    with (HERE / "staged_forms.csv").open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    kotli_rows = [row for row in forms if ":KNA:" in row[10] or ":KTA:" in row[10]]
    assert len(kotli_rows) == 465
    assert {row[0] for row in kotli_rows} == {"Noiri"}
    assert all(row[14].startswith("dialect:Noiri:sil-noira-2015-kotli-") for row in kotli_rows)


def test_dhule_crosswalk_is_complete_but_not_a_reading_authority() -> None:
    rows = read_dicts(HERE / "dhule_republication_reconciliation.tsv", "\t")
    assert len(rows) == 630
    assert len({(row["Item"], row["Noira_Site"]) for row in rows}) == 630
    assert Counter(row["Noira_Site"] for row in rows) == Counter(NAS=210, BMU=210, NTO=210)
    assert Counter(row["Comparison"] for row in rows) == Counter(
        {"literal-ledger-exact": 3, "same-source-representation-differs": 627}
    )
    assert {
        row["Disposition"] for row in rows
    } == {"exclude Noira republication; retain primary ESR 2013-004 route"}
    manifest = json.loads((HERE / "preintegration_manifest.json").read_text(encoding="utf-8"))
    contract = manifest["republication_reconciliation"]["contract"]
    assert "source-team/list identity" in contract
    assert "not lexical disagreements" in contract
    assert "not used to verify a Noira reading" in contract


def test_profile_and_render_contracts_are_complete() -> None:
    profile = read_dicts(HERE / "profile_inventory.tsv", "\t")
    assert len(profile) == 54
    assert all(row["Present_In_Staged_Targets"] == "yes" for row in profile)
    assert sum(int(row["Staged_Input_Occurrences"]) for row in profile) > 0
    renders = read_dicts(HERE / "render_hashes.tsv", "\t")
    assert len(renders) == 46
    assert [int(row["Physical_PDF_Page"]) for row in renders] == list(range(33, 79))
    assert [int(row["Printed_Page"]) for row in renders] == list(range(27, 73))
    assert {(row["Width"], row["Height"], row["DPI"]) for row in renders} == {
        ("1224", "1584", "144")
    }
    assert all("topology-audit-only" in row["Evidence_Class"] for row in renders)
    assert all(len(row["SHA256"]) == 64 for row in renders)


def test_manifest_is_an_integration_ready_zero_blocker_contract() -> None:
    manifest = json.loads((HERE / "preintegration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "source-local-preintegration-audit-complete"
    assert manifest["topology"] == {
        "prompts": 210,
        "lists": 17,
        "conceptual_cells": 3570,
        "new_target_lists": 11,
        "new_target_cells": 2310,
        "republished_dhule_lists": 3,
        "republished_dhule_cells": 630,
        "comparison_control_lists": 3,
        "comparison_control_cells": 630,
    }
    assert manifest["statuses"]["unresolved_coordinates"] == []
    assert manifest["staged_target_forms"]["rows"] == 2714
    assert manifest["staged_target_forms"]["unique_entry_keys"] == 2714
    assert manifest["profile"]["missing_staged_input_sequences"] == []
    contract = manifest["shared_integration_contract"]
    assert contract["install_source_local_target_rows_byte_for_byte"] == 2714
    assert contract["republished_dhule_cells_audit_only"] == 630
    assert contract["comparison_control_cells_audit_only"] == 630
    assert contract["scholarly_identity_blockers"] == []
    assert "Provisionally route Narayanpur and Taradi" in manifest["identity_contract"]["kotli_mapping"]
    assert hashlib.sha256((HERE / "render_hashes.tsv").read_bytes()).hexdigest() == manifest["renders"]["manifest_sha256"]
    assert hashlib.sha256((HERE / "profile_inventory.tsv").read_bytes()).hexdigest() == manifest["profile"]["inventory_sha256"]
