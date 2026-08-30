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


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_preintegration_audit_is_fresh_and_primary_pdf_is_pinned() -> None:
    result = subprocess.run(
        [sys.executable, str(HERE / "preintegration_audit.py")],
        check=True, capture_output=True, text=True,
    )
    assert "cells=3780" in result.stdout
    assert "staged_forms=2100" in result.stdout
    assert "ho_republication=1050" in result.stdout
    assert "unresolved=0" in result.stdout
    assert pre.pdf_metadata() == {
        "path": "tmp/pdfs/bhumij_2015/silesr2015_026.pdf",
        "bytes": 2725577,
        "pages": 130,
        "sha256": pre.PDF_HASH,
        "title": pre.PDF_TITLE,
        "author": pre.PDF_AUTHOR,
        "page_size_points": "612 x 792",
        "page_rotation": 0,
    }


def test_manual_freeze_census_expansion_and_dispositions_are_exhaustive() -> None:
    rows, audit, forms = pre.verify_frozen_package()
    assert len(rows) == len(audit) == 3780
    assert Counter(row["Review_Status"] for row in rows) == Counter(
        attested=3690, source_blank=90
    )
    assert sum(
        len(row["Manual_Transcription"].split(" | "))
        for row in rows if row["Review_Status"] == "attested"
    ) == 3876
    assert len(forms) == len({row["Entry_Key"] for row in forms}) == 2100
    assert Counter(row["Disposition"] for row in audit) == Counter({
        "target-staged": 2054,
        "target-source-blank-excluded": 46,
        "comparison-control-excluded": 1636,
        "comparison-control-blank-excluded": 44,
    })


def test_source_qualifier_is_retained_as_metadata_not_form_text() -> None:
    rows, _, forms = pre.verify_frozen_package()
    source = next(row for row in rows if row["Item"] == "195" and row["Site_Code"] == "LAD")
    assert source["Manual_Transcription"] == "sɛnodʒɑnʌ, dolɑ"
    assert source["Confidence"] == "medium"
    assert source["Uncertainty"] == "source appends '(?)' after dolɑ; printed form itself is legible"
    staged = [row for row in forms if row["Entry_Key"] == "bhumij1989-ladhiramsai-i195-a01"]
    assert len(staged) == 1
    assert staged[0]["Form"] == "sɛnodʒɑnʌ, dolɑ"
    assert "(?)" not in staged[0]["Form"]


def test_target_control_identity_policy_is_exact() -> None:
    registry = pre.verify_identity_policy()
    by_site = {row["Site_Code"]: row for row in registry}
    assert {row["Site_Code"] for row in registry if row["Install"] == "yes"} == pre.TARGETS
    assert {row["Site_Code"] for row in registry if row["Install"] == "no"} == pre.CONTROLS
    assert {row["Dialect_ID"] for row in registry if row["Install"] == "yes"} == set(
        pre.EXPECTED_DIALECTS.values()
    )
    assert by_site["UDA"]["Source_Language_Label"] == "Mundari? Bhumij?"
    assert by_site["UDA"]["Language_ID"] == "unr"
    assert not any(by_site[site]["Dialect_ID"] for site in pre.CONTROLS)


def test_ho_republication_is_exhaustive_post_freeze_comparison_only() -> None:
    rows = pre.verify_overlap()
    assert len(rows) == 1050
    assert Counter(row["Bhumij_Site_Code"] for row in rows) == Counter(
        {site: 210 for site in pre.OVERLAP_TARGETS}
    )
    assert Counter(row["Representation_Comparison"] for row in rows) == Counter({
        "blank-parity": 11,
        "unicode-exact-after-label-removal": 221,
        "publication-transcription-differs": 818,
    })
    manifest = json.loads((HERE / "preintegration_manifest.json").read_text(encoding="utf-8"))
    contract = manifest["ho_2024_republication"]["contract"]
    assert "locality, date, speaker, and recorder" in contract
    assert "post-freeze comparison evidence only" in contract
    assert "did not verify any Bhumij reading" in contract


def test_profile_render_and_manifest_contracts_are_pinned() -> None:
    renders = read_dicts(HERE / "render_hashes.tsv")
    assert len(renders) == 43
    assert [int(row["Physical_PDF_Page"]) for row in renders] == list(range(34, 77))
    assert [int(row["Printed_Page"]) for row in renders] == list(range(29, 72))
    assert {(row["Width"], row["Height"], row["DPI"]) for row in renders} == {
        ("1224", "1584", "144")
    }
    manifest = json.loads((HERE / "preintegration_manifest.json").read_text(encoding="utf-8"))
    assert manifest == pre.expected_manifest()
    assert manifest["profile"]["mapping_rows"] == 72
    assert manifest["profile"]["staged_characters"] == 53
    assert manifest["profile"]["missing_staged_input_sequences"] == []
    assert manifest["shared_integration_contract"]["scholarly_identity_blockers"] == []
    assert manifest["shared_integration_contract"]["unresolved_lexical_coordinates"] == 0
    assert hashlib.sha256((HERE / "render_hashes.tsv").read_bytes()).hexdigest() == (
        manifest["renders"]["manifest_sha256"]
    )
