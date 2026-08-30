from __future__ import annotations

import csv
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
LEDGER = HERE / "manual_chunks" / "items_083_124_hand_keyed.tsv"
spec = importlib.util.spec_from_file_location("dhurwa_guard_p19", HERE / "import_dhurwa_2021.py")
guard = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(guard)


def load_p19() -> list[dict[str, str]]:
    return guard.load_manual_cells(LEDGER)


def test_p19_is_exhaustive_ocr_blind_declared_and_nfc():
    rows = load_p19()
    assert len(rows) == 42 * 5 == 210
    assert {int(row["Item"]) for row in rows} == set(range(83, 125))
    assert {row["Site_Code"] for row in rows} == {"TIR", "NET", "DHA", "KUK", "U5"}
    assert all(row["PDF_Page"] == "19" and row["Printed_Page"] == "14" for row in rows)
    assert all(row["Reviewer_Declaration"] == guard.DECLARATION for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        fields = csv.DictReader(handle, delimiter="\t").fieldnames or []
    assert not any("ocr" in field.casefold() for field in fields)


def test_p19_exact_status_and_response_accounting():
    forms, audit, counts = guard.build_checkpoint(load_p19())
    assert counts == {
        "reviewed_cells": 210,
        "attested_cells": 210,
        "source_blank_cells": 0,
        "ambiguous_cells": 0,
        "illegible_cells": 0,
        "expanded_responses": 211,
        "known_target_cells": 168,
        "known_target_forms": 169,
        "unresolved_identity_cells": 42,
        "unresolved_identity_responses": 42,
    }
    assert len(forms) == 169 and len(audit) == 210
    assert len({row[10] for row in forms}) == 169


def test_p19_visual_rechecks_are_diplomatic():
    rows = load_p19()
    by_key = {(row["Item"], row["Site_Code"]): row["Manual_Transcription"] for row in rows}
    assert by_key[("89", "TIR")] == "cɪr"
    assert by_key[("89", "U5")] == "ciru"
    assert by_key[("91", "DHA")] == "neŋgɖa"
    assert by_key[("96", "TIR")] == "nurñyi"
    assert by_key[("96", "U5")] == "urñdʒil"
    assert by_key[("108", "TIR")] == "cinɖu"
    assert by_key[("108", "DHA")] == "ciɳɖ"
    assert by_key[("118", "TIR")] == "ʈɪʈ:e ɖɛlkul"
    assert by_key[("118", "DHA")] == "ʈɪʈ:eɖɛlkul"
    assert by_key[("118", "U5")] == "ʈɛlkul po:kal"
    assert by_key[("123", "TIR")] == "pɪnge"
    assert by_key[("123", "KUK")] == "pɪɖne"
    assert by_key[("124", "NET")] == "ad ɖina"
    assert by_key[("124", "U5")] == "aɖ ɖina"


def test_p19_expands_only_the_printed_slash():
    rows = load_p19()
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert by_key[("105", "KUK")]["Manual_Transcription"] == "iya/ʈʌl"
    assert guard.expand_cell(by_key[("105", "KUK")]["Manual_Transcription"]) == ["iya", "ʈʌl"]
    assert guard.expand_cell(by_key[("118", "TIR")]["Manual_Transcription"]) == ["ʈɪʈ:e ɖɛlkul"]
    assert guard.expand_cell(by_key[("124", "TIR")]["Manual_Transcription"]) == ["aʈ ɖina"]


def test_p19_blank_header_u5_is_audited_but_never_staged():
    forms, audit, _ = guard.build_checkpoint(load_p19())
    assert all(":U5:" not in row[10] for row in forms)
    u5_audit = [row for row in audit if row["Site_Code"] == "U5"]
    assert len(u5_audit) == 42
    assert all(row["Installed_Count"] == "0" for row in u5_audit)
    assert all("identity unresolved" in row["Disposition"] for row in u5_audit)


def test_p19_guard_rejects_ocr_bearing_schema(tmp_path: Path):
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


def test_current_cumulative_checkpoint_manifest_and_profile_are_exact():
    rows = guard.load_all_manual_cells()
    forms, audit, counts = guard.build_checkpoint(rows)
    assert len(rows) == 1000 and len(forms) == 809 and len(audit) == 1000
    assert counts["reviewed_cells"] == 1000
    assert counts["attested_cells"] == 995
    assert counts["source_blank_cells"] == 5
    assert counts["expanded_responses"] == 1008
    assert counts["known_target_forms"] == 809
    assert counts["unresolved_identity_cells"] == 200
    assert counts["unresolved_identity_responses"] == 199
    profile = guard.load_profile()
    assert all("�" not in guard.convert(row[2], profile) for row in forms)

    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    checkpoint = manifest["manual_review_checkpoint"]
    assert checkpoint["completed_items"] == "1-200"
    assert checkpoint["remaining_cells"] == 0
    assert checkpoint["reviewed_cells"] == 1000
    assert checkpoint["known_target_forms"] == 809
