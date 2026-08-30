import csv
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_190_210_hand_keyed.tsv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("noira_guard_190_210", PACKAGE / "import_noira_2015.py")


def response_count(row):
    if row["Review_Status"] != "attested":
        return 0
    return len(row["Manual_Transcription"].split(" | "))


def test_complete_ocr_blind_final_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 21 * 17 == 357
    assert {int(row["Item"]) for row in rows} == set(range(190, 211))
    assert len({(row["Item"], row["Site_Code"]) for row in rows}) == 357
    assert {row["Reviewer_Declaration"] for row in rows} == {guard.DECLARATION}
    assert all("OCR" not in field.upper() for field in rows[0])
    assert all(
        unicodedata.is_normalized("NFC", value)
        for row in rows for value in row.values()
    )


def test_final_chunk_accounting():
    rows = guard.load_manual_cells(LEDGER)
    controls = {"GUJ", "MAR", "HIN"}
    assert sum(row["Review_Status"] == "attested" for row in rows) == 357
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 0
    assert sum(row["Review_Status"] in {"ambiguous", "illegible"} for row in rows) == 0
    assert sum(response_count(row) for row in rows) == 489
    assert sum(response_count(row) for row in rows if row["Site_Code"] in controls) == 71
    assert len(guard.stage_target_forms(rows)) == 418


def test_final_page_continuations_are_explicit():
    rows = guard.load_manual_cells(LEDGER)
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    expected = {
        ("191", "HIN"): ("74", "left"),
        ("191", "NTE"): ("74", "right"),
        ("193", "DBM"): ("74", "right"),
        ("193", "DBA"): ("75", "left"),
        ("197", "BMU"): ("75", "right"),
        ("197", "DBM"): ("76", "left"),
        ("201", "KTA"): ("76", "right"),
        ("201", "GTA"): ("77", "left"),
        ("206", "MAR"): ("77", "right"),
        ("206", "HIN"): ("78", "left"),
    }
    for key, coordinate in expected.items():
        assert (by_key[key]["PDF_Page"], by_key[key]["Column"]) == coordinate
        assert int(by_key[key]["Printed_Page"]) == int(by_key[key]["PDF_Page"]) - 6


def test_difficult_900_dpi_readings_are_diplomatic():
    rows = guard.load_manual_cells(LEDGER)
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    expected = {
        ("190", "NAS"): "apiho",
        ("195", "NGO"): "sʌnɑ | sʌna",
        ("195", "DBM"): "sɑni | sɑninu",
        ("200", "DBA"): "hamble | hambəlja",
        ("200", "GTA"): "sabʌl̪",
        ("202", "GTA"): "hũ",
        ("208", "DBA"): "amɪ",
    }
    for key, transcription in expected.items():
        assert by_key[key]["Manual_Transcription"] == transcription
    assert by_key[("202", "GTA")]["Source_Cognate_Labels"] == "?"


def test_guard_rejects_ocr_bearing_or_undeclared_rows(tmp_path):
    rows = list(csv.DictReader(LEDGER.open(encoding="utf-8"), delimiter="\t"))
    bad_ocr = tmp_path / "bad_ocr.tsv"
    fields = list(rows[0]) + ["OCR_Evidence"]
    with bad_ocr.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows([{**row, "OCR_Evidence": "inadmissible"} for row in rows])
    with pytest.raises(AssertionError):
        guard.load_manual_cells(bad_ocr)

    bad_declaration = tmp_path / "bad_declaration.tsv"
    rows[0]["Reviewer_Declaration"] = ""
    with bad_declaration.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(AssertionError):
        guard.load_manual_cells(bad_declaration)


def test_exhaustive_package_accounting_and_profile():
    rows = guard.load_all_manual_cells()
    registry = guard.load_registry()
    forms, audit, counts = guard.build_package(rows, registry)
    guard.validate_profile(forms)
    reconciliation = guard.build_republication_audit(rows)
    assert len(rows) == len(audit) == 3570
    assert len(forms) == counts["installed_forms"] == 2714
    assert counts["attested_cells"] == 3526
    assert counts["source_blank_cells"] == 44
    assert counts["expanded_responses"] == 4385
    assert counts["new_target_attested_cells"] == 2271
    assert counts["new_target_blank_cells"] == 39
    assert counts["republished_dhule_responses_excluded"] == 834
    assert counts["control_responses_excluded"] == 837
    assert len(reconciliation) == 630
    assert {row["Disposition"] for row in reconciliation} == {
        "exclude Noira republication; retain primary ESR 2013-004 route"
    }


def test_written_artifacts_and_manifest_match_rebuild():
    rows = guard.load_all_manual_cells()
    forms, audit, counts = guard.build_package(rows, guard.load_registry())
    with guard.STAGED_FORMS.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == forms
    with guard.STAGED_AUDIT.open(encoding="utf-8", newline="") as handle:
        written_audit = list(csv.DictReader(handle, delimiter="\t"))
    assert written_audit == audit
    manifest = json.loads(guard.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["manual_review"]["remaining_cells"] == 0
    for key, value in counts.items():
        assert manifest["manual_review"][key] == value
    for artifact in manifest["artifacts"].values():
        path = PACKAGE / artifact["path"]
        assert guard.sha256(path) == artifact["sha256"]
