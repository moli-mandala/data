import csv
import importlib.util
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_109_135_hand_keyed.tsv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("noira_guard_109_135", PACKAGE / "import_noira_2015.py")


def response_count(row):
    if row["Review_Status"] != "attested":
        return 0
    return len(row["Manual_Transcription"].split(" | "))


def test_complete_ocr_blind_manual_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 27 * 17 == 459
    assert {int(r["Item"]) for r in rows} == set(range(109, 136))
    assert len({(r["Item"], r["Site_Code"]) for r in rows}) == 459
    assert {r["Reviewer_Declaration"] for r in rows} == {guard.DECLARATION}
    assert all("OCR" not in key.upper() for key in rows[0])
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())


def test_accounting_and_target_staging():
    rows = guard.load_manual_cells(LEDGER)
    controls = {"GUJ", "MAR", "HIN"}
    targets = [r for r in rows if r["Site_Code"] not in controls]
    control_rows = [r for r in rows if r["Site_Code"] in controls]
    assert sum(r["Review_Status"] == "attested" for r in rows) == 457
    assert sum(r["Review_Status"] == "source_blank" for r in rows) == 2
    assert sum(r["Review_Status"] in {"ambiguous", "illegible"} for r in rows) == 0
    assert sum(response_count(r) for r in rows) == 580
    assert len(targets) == 378
    assert sum(response_count(r) for r in targets) == 457
    assert len(control_rows) == 81
    assert sum(response_count(r) for r in control_rows) == 123
    assert len(guard.stage_target_forms(rows)) == 457


def test_page_and_column_continuations_are_explicit():
    rows = guard.load_manual_cells(LEDGER)
    by_key = {(r["Item"], r["Site_Code"]): r for r in rows}
    expected = {
        ("109", "NCH"): ("56", "left"),
        ("109", "DBM"): ("56", "right"),
        ("110", "NTE"): ("56", "right"),
        ("110", "TKO"): ("57", "left"),
        ("117", "DBA"): ("58", "left"),
        ("117", "NTO"): ("58", "right"),
        ("119", "NTE"): ("58", "right"),
        ("119", "TKO"): ("59", "left"),
        ("124", "GUJ"): ("59", "right"),
        ("124", "MAR"): ("60", "left"),
        ("129", "DBM"): ("60", "right"),
        ("129", "DBA"): ("61", "left"),
        ("135", "HIN"): ("62", "left"),
        ("135", "NTE"): ("62", "right"),
    }
    for key, coordinate in expected.items():
        assert (by_key[key]["PDF_Page"], by_key[key]["Column"]) == coordinate
        assert int(by_key[key]["Printed_Page"]) == int(by_key[key]["PDF_Page"]) - 6


def test_source_blanks_are_exact_and_not_staged():
    rows = guard.load_manual_cells(LEDGER)
    blanks = [r for r in rows if r["Review_Status"] == "source_blank"]
    assert {(r["Item"], r["Site_Code"]) for r in blanks} == {
        ("115", "DBA"),
        ("116", "DBA"),
    }
    for row in blanks:
        assert (row["PDF_Page"], row["Printed_Page"], row["Column"]) == (
            "58", "52", "left"
        )
        assert row["Manual_Transcription"] == ""
        assert row["Source_Cognate_Labels"] == ""
        assert row["Uncertainty"] == "source explicitly prints '0 no entry'"


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
        HERE / "items_082_108_hand_keyed.tsv",
        HERE / "items_109_135_hand_keyed.tsv",
    ]
    rows = [row for ledger in ledgers for row in guard.load_manual_cells(ledger)]
    assert len(rows) == 135 * 17 == 2295
    assert len({(r["Item"], r["Site_Code"]) for r in rows}) == 2295
    assert {int(r["Item"]) for r in rows} == set(range(1, 136))
    assert sum(r["Review_Status"] == "attested" for r in rows) == 2257
    assert sum(r["Review_Status"] == "source_blank" for r in rows) == 38
    assert sum(response_count(r) for r in rows) == 2797
    assert len(guard.stage_target_forms(rows)) == 2224
