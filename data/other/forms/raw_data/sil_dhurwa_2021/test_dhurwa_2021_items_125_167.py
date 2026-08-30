from __future__ import annotations

import csv
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
LEDGER = HERE / "manual_chunks" / "items_125_167_hand_keyed.tsv"
spec = importlib.util.spec_from_file_location("dhurwa_guard_p20", HERE / "import_dhurwa_2021.py")
guard = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(guard)


def load_p20() -> list[dict[str, str]]:
    return guard.load_manual_cells(LEDGER)


def test_p20_is_exhaustive_ocr_blind_declared_and_nfc():
    rows = load_p20()
    assert len(rows) == 43 * 5 == 215
    assert {int(row["Item"]) for row in rows} == set(range(125, 168))
    assert {row["Site_Code"] for row in rows} == {"TIR", "NET", "DHA", "KUK", "U5"}
    assert all(row["PDF_Page"] == "20" and row["Printed_Page"] == "15" for row in rows)
    assert all(row["Reviewer_Declaration"] == guard.DECLARATION for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    with LEDGER.open(encoding="utf-8", newline="") as handle:
        fields = csv.DictReader(handle, delimiter="\t").fieldnames or []
    assert not any("ocr" in field.casefold() for field in fields)


def test_p20_exact_status_and_response_accounting():
    forms, audit, counts = guard.build_checkpoint(load_p20())
    assert counts == {
        "reviewed_cells": 215,
        "attested_cells": 212,
        "source_blank_cells": 3,
        "ambiguous_cells": 0,
        "illegible_cells": 0,
        "expanded_responses": 215,
        "known_target_cells": 172,
        "known_target_forms": 172,
        "unresolved_identity_cells": 43,
        "unresolved_identity_responses": 43,
    }
    assert len(forms) == 172 and len(audit) == 215
    assert len({row[10] for row in forms}) == 172


def test_p20_source_blanks_are_exact_and_diplomatic():
    rows = load_p20()
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {("149", "DHA"), ("162", "KUK"), ("162", "U5")}
    for row in rows:
        if row["Review_Status"] == "source_blank":
            assert row["Manual_Transcription"] == ""
            assert row["Uncertainty"] == "source prints double hyphen"


def test_p20_visual_rechecks_are_diplomatic():
    rows = load_p20()
    by_key = {(row["Item"], row["Site_Code"]): row["Manual_Transcription"] for row in rows}
    assert by_key[("125", "TIR")] == "nɛlɪŋ"
    assert by_key[("125", "NET")] == "nɛliŋ"
    assert by_key[("128", "U5")] == "dʒeʈa/neɳɖ"
    assert by_key[("133", "DHA")] == "veʈʌraro"
    assert by_key[("138", "KUK")] == "ʈinʈa kɛy"
    assert by_key[("140", "U5")] == "lʌk:nɖi"
    assert by_key[("143", "DHA")] == "pɪʈiʈʌ"
    assert by_key[("144", "KUK")] == "poɖɪ"
    assert by_key[("146", "TIR")] == "bɪl:oʈ"
    assert by_key[("152", "TIR")] == "ɪrɛɖuk"
    assert by_key[("153", "DHA")] == "mu:ɳɖu:k"
    assert by_key[("163", "DHA")] == "ceɳɖ ko:ɖu"
    assert by_key[("167", "U5")] == "bɪlag bɪlag"


def test_p20_expands_only_printed_slashes():
    rows = load_p20()
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert guard.expand_cell(by_key[("128", "U5")]["Manual_Transcription"]) == ["dʒeʈa", "neɳɖ"]
    for site in ["DHA", "KUK"]:
        assert guard.expand_cell(by_key[("165", site)]["Manual_Transcription"]) == ["cɪl:a", "ɛra"]
    assert guard.expand_cell(by_key[("163", "TIR")]["Manual_Transcription"]) == ["cenɖ ko:l"]


def test_p20_blank_header_u5_is_audited_but_never_staged():
    forms, audit, _ = guard.build_checkpoint(load_p20())
    assert all(":U5:" not in row[10] for row in forms)
    u5_audit = [row for row in audit if row["Site_Code"] == "U5"]
    assert len(u5_audit) == 43
    assert all(row["Installed_Count"] == "0" for row in u5_audit)
    assert all("identity unresolved" in row["Disposition"] for row in u5_audit)


def test_p20_guard_rejects_ocr_bearing_schema(tmp_path: Path):
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
