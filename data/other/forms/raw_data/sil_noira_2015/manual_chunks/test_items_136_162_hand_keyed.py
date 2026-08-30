import csv
import importlib.util
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_136_162_hand_keyed.tsv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("noira_guard_136_162", PACKAGE / "import_noira_2015.py")


def response_count(row):
    if row["Review_Status"] != "attested":
        return 0
    return len(row["Manual_Transcription"].split(" | "))


def test_complete_ocr_blind_manual_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 27 * 17 == 459
    assert {int(r["Item"]) for r in rows} == set(range(136, 163))
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
    assert sum(response_count(r) for r in rows) == 492
    assert len(targets) == 378
    assert sum(response_count(r) for r in targets) == 402
    assert len(control_rows) == 81
    assert sum(response_count(r) for r in control_rows) == 90
    assert len(guard.stage_target_forms(rows)) == 402


def test_page_and_column_continuations_are_explicit():
    rows = guard.load_manual_cells(LEDGER)
    by_key = {(r["Item"], r["Site_Code"]): r for r in rows}
    expected = {
        ("136", "NCH"): ("62", "right"),
        ("138", "DBA"): ("62", "right"),
        ("138", "NTO"): ("63", "left"),
        ("143", "NTO"): ("63", "right"),
        ("143", "KNA"): ("64", "left"),
        ("145", "TKO"): ("64", "left"),
        ("145", "NJA"): ("64", "right"),
        ("148", "GTA"): ("64", "right"),
        ("148", "GUJ"): ("65", "left"),
        ("151", "DBA"): ("65", "left"),
        ("151", "NTO"): ("65", "right"),
        ("154", "BMU"): ("65", "right"),
        ("154", "DBM"): ("66", "left"),
        ("159", "NTE"): ("66", "right"),
        ("159", "TKO"): ("67", "left"),
        ("162", "KNA"): ("67", "left"),
        ("162", "KTA"): ("67", "right"),
    }
    for key, coordinate in expected.items():
        assert (by_key[key]["PDF_Page"], by_key[key]["Column"]) == coordinate
        assert int(by_key[key]["Printed_Page"]) == int(by_key[key]["PDF_Page"]) - 6


def test_difficult_900_dpi_readings_are_diplomatic():
    rows = guard.load_manual_cells(LEDGER)
    by_key = {(r["Item"], r["Site_Code"]): r for r in rows}
    expected = {
        ("136", "MAR"): "uʂɳʌ | gʌrʌm",
        ("137", "NTO"): "heɭo",
        ("140", "MAR"): "dzʌwʌɭ",
        ("144", "MAR"): "bhari | wʌdzʌɳɖar | dzuɖ",
        ("148", "HIN"): "səɸeɖ",
        ("149", "NTE"): "kẽɳdʒe",
        ("154", "NCH"): "tʒjʌr",
        ("161", "BMU"): "igjaʌ | igjara",
        ("161", "DBA"): "igijʌre",
    }
    for key, form in expected.items():
        assert by_key[key]["Manual_Transcription"] == form


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


def test_cumulative_chunks_through_162_are_disjoint():
    ledgers = [
        HERE / "items_001_027_hand_keyed.tsv",
        HERE / "items_028_054_hand_keyed.tsv",
        HERE / "items_055_081_hand_keyed.tsv",
        HERE / "items_082_108_hand_keyed.tsv",
        HERE / "items_109_135_hand_keyed.tsv",
        HERE / "items_136_162_hand_keyed.tsv",
    ]
    rows = [row for ledger in ledgers for row in guard.load_manual_cells(ledger)]
    assert len(rows) == 162 * 17 == 2754
    assert len({(r["Item"], r["Site_Code"]) for r in rows}) == 2754
    assert {int(r["Item"]) for r in rows} == set(range(1, 163))
    assert sum(r["Review_Status"] == "attested" for r in rows) == 2716
    assert sum(r["Review_Status"] == "source_blank" for r in rows) == 38
    assert sum(response_count(r) for r in rows) == 3289
    assert len(guard.stage_target_forms(rows)) == 2626
