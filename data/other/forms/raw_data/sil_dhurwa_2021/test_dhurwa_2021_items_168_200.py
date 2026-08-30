from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
LEDGER = HERE / "manual_chunks" / "items_168_200_hand_keyed.tsv"
spec = importlib.util.spec_from_file_location("dhurwa_guard_p21", HERE / "import_dhurwa_2021.py")
guard = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(guard)


def load_p21() -> list[dict[str, str]]:
    return guard.load_manual_cells(LEDGER)


def test_p21_is_exhaustive_ocr_blind_declared_and_nfc():
    rows = load_p21()
    assert len(rows) == 33 * 5 == 165
    assert {int(row["Item"]) for row in rows} == set(range(168, 201))
    assert {row["Site_Code"] for row in rows} == {"TIR", "NET", "DHA", "KUK", "U5"}
    assert all(row["PDF_Page"] == "21" and row["Printed_Page"] == "16" for row in rows)
    assert all(row["Reviewer_Declaration"] == guard.DECLARATION for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        fields = csv.DictReader(handle, delimiter="\t").fieldnames or []
    assert not any("ocr" in field.casefold() for field in fields)


def test_p21_exact_status_and_response_accounting():
    forms, audit, counts = guard.build_checkpoint(load_p21())
    assert counts == {
        "reviewed_cells": 165,
        "attested_cells": 165,
        "source_blank_cells": 0,
        "ambiguous_cells": 0,
        "illegible_cells": 0,
        "expanded_responses": 168,
        "known_target_cells": 132,
        "known_target_forms": 135,
        "unresolved_identity_cells": 33,
        "unresolved_identity_responses": 33,
    }
    assert len(forms) == 135 and len(audit) == 165
    assert len({row[10] for row in forms}) == 135


def test_p21_has_no_source_blanks_or_unresolved_transcriptions():
    rows = load_p21()
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert all(row["Manual_Transcription"] for row in rows)
    assert all(not row["Uncertainty"] for row in rows)


def test_p21_difficult_visual_readings_are_diplomatic():
    rows = load_p21()
    by_key = {(row["Item"], row["Site_Code"]): row["Manual_Transcription"] for row in rows}
    assert by_key[("169", "TIR")] == "mañ"
    assert by_key[("169", "KUK")] == "mʌŋ:u"
    assert by_key[("171", "NET")] == "bʌŋdaŋ/ʈʌl"
    assert by_key[("174", "DHA")] == "meɳɖir"
    assert by_key[("176", "U5")] == "uɖen puyɪl"
    assert by_key[("179", "DHA")] == "ʈɛlʈa kull/kadrel"
    assert by_key[("180", "KUK")] == "mɛɖer"
    assert by_key[("180", "U5")] == "mɛɖek"
    assert by_key[("189", "TIR")] == "nɛɳɖɪl"
    assert by_key[("195", "DHA")] == "puɖu:ʈ"
    assert by_key[("199", "TIR")] == "kɛridʒ"


def test_p21_expands_only_three_printed_slashes():
    rows = load_p21()
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    expected = {
        ("171", "NET"): ["bʌŋdaŋ", "ʈʌl"],
        ("179", "DHA"): ["ʈɛlʈa kull", "kadrel"],
        ("189", "NET"): ["nɛɳɖɪl", "neli"],
    }
    observed = {
        key: guard.expand_cell(row["Manual_Transcription"])
        for key, row in by_key.items()
        if len(guard.expand_cell(row["Manual_Transcription"])) > 1
    }
    assert observed == expected


def test_p21_blank_header_u5_is_audited_but_never_staged():
    forms, audit, _ = guard.build_checkpoint(load_p21())
    assert all(":U5:" not in row[10] for row in forms)
    u5_audit = [row for row in audit if row["Site_Code"] == "U5"]
    assert len(u5_audit) == 33
    assert all(row["Installed_Count"] == "0" for row in u5_audit)
    assert all("identity unresolved" in row["Disposition"] for row in u5_audit)


def test_p21_guard_rejects_ocr_bearing_schema(tmp_path: Path):
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    bad = tmp_path / "ocr_bearing.tsv"
    fields = list(rows[0]) + ["OCR_Evidence"]
    with bad.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{**row, "OCR_Evidence": "inadmissible"} for row in rows])
    with pytest.raises(AssertionError):
        guard.load_manual_cells(bad)


def test_complete_source_counts_profile_manifest_and_hashes_are_exact():
    rows = guard.load_all_manual_cells()
    forms, audit, counts = guard.build_checkpoint(rows)
    assert len(rows) == 1000 and len(forms) == 809 and len(audit) == 1000
    assert counts == {
        "reviewed_cells": 1000,
        "attested_cells": 995,
        "source_blank_cells": 5,
        "ambiguous_cells": 0,
        "illegible_cells": 0,
        "expanded_responses": 1008,
        "known_target_cells": 800,
        "known_target_forms": 809,
        "unresolved_identity_cells": 200,
        "unresolved_identity_responses": 199,
    }
    profile = guard.load_profile()
    assert all("�" not in guard.convert(row[2], profile) for row in forms)

    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    checkpoint = manifest["manual_review_checkpoint"]
    assert checkpoint["completed_items"] == "1-200"
    assert checkpoint["remaining_items"] == "none"
    assert checkpoint["remaining_cells"] == 0
    assert checkpoint["reviewed_cells"] == 1000
    assert checkpoint["known_target_forms"] == 809
    for name, artifact in manifest["artifacts"].items():
        artifacts = artifact if name == "manual_ledgers" else [artifact]
        for item in artifacts:
            path = HERE / item["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
