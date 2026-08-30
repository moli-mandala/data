from __future__ import annotations

import csv
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
LEDGER = HERE / "manual_chunks" / "items_042_082_hand_keyed.tsv"
spec = importlib.util.spec_from_file_location("dhurwa_guard_p18", HERE / "import_dhurwa_2021.py")
guard = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(guard)


def load_p18() -> list[dict[str, str]]:
    return guard.load_manual_cells(LEDGER)


def test_p18_is_exhaustive_ocr_blind_declared_and_nfc():
    rows = load_p18()
    assert len(rows) == 41 * 5 == 205
    assert {int(row["Item"]) for row in rows} == set(range(42, 83))
    assert {row["Site_Code"] for row in rows} == {"TIR", "NET", "DHA", "KUK", "U5"}
    assert all(row["PDF_Page"] == "18" and row["Printed_Page"] == "13" for row in rows)
    assert all(row["Reviewer_Declaration"] == guard.DECLARATION for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        fields = csv.DictReader(handle, delimiter="\t").fieldnames or []
    assert not any("ocr" in field.casefold() for field in fields)


def test_p18_exact_status_and_response_accounting():
    forms, audit, counts = guard.build_checkpoint(load_p18())
    assert counts == {
        "reviewed_cells": 205,
        "attested_cells": 205,
        "source_blank_cells": 0,
        "ambiguous_cells": 0,
        "illegible_cells": 0,
        "expanded_responses": 210,
        "known_target_cells": 164,
        "known_target_forms": 169,
        "unresolved_identity_cells": 41,
        "unresolved_identity_responses": 41,
    }
    assert len(forms) == 169 and len(audit) == 205
    assert len({row[10] for row in forms}) == 169


def test_p18_visual_rechecks_and_printed_variants_are_diplomatic():
    rows = load_p18()
    by_key = {(row["Item"], row["Site_Code"]): row["Manual_Transcription"] for row in rows}
    assert by_key[("46", "KUK")] == "vaya/kʌmo"
    assert by_key[("47", "KUK")] == "kop:a/ke:ɳɖi"
    assert by_key[("55", "NET")] == "ʈuri/guɳɖa"
    assert by_key[("58", "KUK")] == "caɳʈi"
    assert by_key[("68", "KUK")] == "cupar"
    assert by_key[("71", "TIR")] == "vɛbɛc:iɖ"
    assert by_key[("75", "TIR")] == "kakaɳɖi | maɖɖu baŋga"
    assert by_key[("78", "TIR")] == "korul:i | lʌsu:n"
    assert by_key[("79", "U5")] == "go:ɳɖri ul:i"


def test_p18_expansion_follows_only_explicit_source_separators():
    rows = load_p18()
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert guard.expand_cell(by_key[("46", "KUK")]["Manual_Transcription"]) == ["vaya", "kʌmo"]
    assert guard.expand_cell(by_key[("75", "TIR")]["Manual_Transcription"]) == ["kakaɳɖi", "maɖɖu baŋga"]
    assert guard.expand_cell(by_key[("78", "TIR")]["Manual_Transcription"]) == ["korul:i", "lʌsu:n"]
    assert guard.expand_cell(by_key[("79", "U5")]["Manual_Transcription"]) == ["go:ɳɖri ul:i"]


def test_p18_blank_header_u5_is_audited_but_never_staged():
    registry = guard.load_registry()
    assert registry["U5"]["Printed_Header"] == ""
    assert registry["U5"]["Scope"] == "unresolved_list_identity"
    assert registry["U5"]["Install"] == "no"
    forms, audit, _ = guard.build_checkpoint(load_p18())
    assert all(":U5:" not in row[10] for row in forms)
    u5_audit = [row for row in audit if row["Site_Code"] == "U5"]
    assert len(u5_audit) == 41
    assert all("identity unresolved" in row["Disposition"] for row in u5_audit)


def test_p18_guard_rejects_ocr_bearing_schema(tmp_path: Path):
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


def test_cumulative_checkpoint_and_profile_are_exact():
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
