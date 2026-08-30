import csv
import importlib.util
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_055_081_hand_keyed.tsv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("noira_guard_055_081", PACKAGE / "import_noira_2015.py")


def test_complete_ocr_blind_manual_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 27 * 17 == 459
    assert {int(r["Item"]) for r in rows} == set(range(55, 82))
    assert len({(r["Item"], r["Site_Code"]) for r in rows}) == 459
    assert {r["Reviewer_Declaration"] for r in rows} == {guard.DECLARATION}
    assert all("OCR" not in key.upper() for key in rows[0])
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())


def test_accounting_and_target_staging():
    rows = guard.load_manual_cells(LEDGER)
    controls = {"GUJ", "MAR", "HIN"}
    targets = [r for r in rows if r["Site_Code"] not in controls]
    control_rows = [r for r in rows if r["Site_Code"] in controls]
    assert sum(r["Review_Status"] == "attested" for r in rows) == 459
    assert sum(r["Review_Status"] == "source_blank" for r in rows) == 0
    assert sum(r["Review_Status"] in {"ambiguous", "illegible"} for r in rows) == 0
    assert sum(len(r["Manual_Transcription"].split(" | ")) for r in rows) == 574
    assert len(targets) == 378
    assert sum(len(r["Manual_Transcription"].split(" | ")) for r in targets) == 463
    assert len(control_rows) == 81
    assert sum(len(r["Manual_Transcription"].split(" | ")) for r in control_rows) == 111
    assert len(guard.stage_target_forms(rows)) == 463


def test_page_and_column_continuations_are_explicit():
    rows = guard.load_manual_cells(LEDGER)
    by_key = {(r["Item"], r["Site_Code"]): r for r in rows}
    expected = {
        ("56", "NGO"): ("44", "right"),
        ("56", "BMU"): ("45", "left"),
        ("60", "NTE"): ("45", "right"),
        ("60", "TKO"): ("46", "left"),
        ("66", "NPN"): ("46", "right"),
        ("66", "NAS"): ("47", "left"),
        ("69", "NTE"): ("47", "right"),
        ("69", "TKO"): ("48", "left"),
        ("74", "GUJ"): ("48", "right"),
        ("74", "MAR"): ("49", "left"),
        ("79", "DBA"): ("49", "right"),
        ("79", "NTO"): ("50", "left"),
        ("58", "DBM"): ("45", "left"),
        ("58", "DBA"): ("45", "right"),
    }
    for key, coordinate in expected.items():
        assert (by_key[key]["PDF_Page"], by_key[key]["Column"]) == coordinate


def test_guard_rejects_ocr_bearing_or_undeclared_rows(tmp_path):
    rows = list(csv.DictReader(LEDGER.open(encoding="utf-8"), delimiter="\t"))

    bad_ocr = tmp_path / "bad_ocr.tsv"
    fieldnames = list(rows[0]) + ["OCR_Evidence"]
    with bad_ocr.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows([{**row, "OCR_Evidence": "must never be accepted"} for row in rows])
    with pytest.raises(AssertionError):
        guard.load_manual_cells(bad_ocr)

    bad_declaration = tmp_path / "bad_declaration.tsv"
    rows[0]["Reviewer_Declaration"] = ""
    with bad_declaration.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(AssertionError):
        guard.load_manual_cells(bad_declaration)


def test_historical_cumulative_chunks_are_disjoint():
    ledgers = [
        HERE / "items_001_027_hand_keyed.tsv",
        HERE / "items_028_054_hand_keyed.tsv",
        HERE / "items_055_081_hand_keyed.tsv",
    ]
    rows = [row for ledger in ledgers for row in guard.load_manual_cells(ledger)]
    assert len(rows) == 81 * 17 == 1377
    assert len({(r["Item"], r["Site_Code"]) for r in rows}) == 1377
    assert {int(r["Item"]) for r in rows} == set(range(1, 82))
    assert sum(r["Review_Status"] == "attested" for r in rows) == 1341
    assert sum(r["Review_Status"] == "source_blank" for r in rows) == 36
    assert sum(
        len(r["Manual_Transcription"].split(" | "))
        for r in rows if r["Review_Status"] == "attested"
    ) == 1660
    assert len(guard.stage_target_forms(rows)) == 1313
