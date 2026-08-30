import csv
import importlib.util
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_028_054_hand_keyed.tsv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("noira_guard_028_054", PACKAGE / "import_noira_2015.py")


def test_complete_ocr_blind_manual_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 27 * 17 == 459
    assert {int(r["Item"]) for r in rows} == set(range(28, 55))
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
    assert sum(len(r["Manual_Transcription"].split(" | ")) for r in rows) == 598
    assert len(targets) == 378
    assert sum(len(r["Manual_Transcription"].split(" | ")) for r in targets) == 471
    assert len(control_rows) == 81
    assert sum(len(r["Manual_Transcription"].split(" | ")) for r in control_rows) == 127
    assert len(guard.stage_target_forms(rows)) == 471


def test_page_and_column_continuations_are_explicit():
    rows = guard.load_manual_cells(LEDGER)
    by_key = {(r["Item"], r["Site_Code"]): r for r in rows}
    assert (by_key[("30", "BMU")]["PDF_Page"], by_key[("30", "BMU")]["Column"]) == ("38", "right")
    assert (by_key[("30", "DBM")]["PDF_Page"], by_key[("30", "DBM")]["Column"]) == ("39", "left")
    assert (by_key[("34", "DBM")]["PDF_Page"], by_key[("34", "DBM")]["Column"]) == ("39", "right")
    assert (by_key[("34", "DBA")]["PDF_Page"], by_key[("34", "DBA")]["Column"]) == ("40", "left")
    assert (by_key[("42", "TKO")]["PDF_Page"], by_key[("42", "TKO")]["Column"]) == ("41", "right")
    assert (by_key[("42", "NJA")]["PDF_Page"], by_key[("42", "NJA")]["Column"]) == ("42", "left")
    assert (by_key[("47", "NCH")]["PDF_Page"], by_key[("47", "NCH")]["Column"]) == ("42", "right")
    assert (by_key[("47", "NPN")]["PDF_Page"], by_key[("47", "NPN")]["Column"]) == ("43", "left")


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
