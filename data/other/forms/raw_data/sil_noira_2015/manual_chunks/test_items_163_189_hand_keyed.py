import csv
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_163_189_hand_keyed.tsv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("noira_guard_163_189", PACKAGE / "import_noira_2015.py")


def response_count(row):
    if row["Review_Status"] != "attested":
        return 0
    return len(row["Manual_Transcription"].split(" | "))


def test_complete_ocr_blind_manual_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 27 * 17 == 459
    assert {int(r["Item"]) for r in rows} == set(range(163, 190))
    assert len({(r["Item"], r["Site_Code"]) for r in rows}) == 459
    assert {r["Reviewer_Declaration"] for r in rows} == {guard.DECLARATION}
    assert all("OCR" not in key.upper() for key in rows[0])
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())


def test_accounting_and_target_staging():
    rows = guard.load_manual_cells(LEDGER)
    controls = {"GUJ", "MAR", "HIN"}
    targets = [r for r in rows if r["Site_Code"] not in controls]
    control_rows = [r for r in rows if r["Site_Code"] in controls]
    assert sum(r["Review_Status"] == "attested" for r in rows) == 453
    assert sum(r["Review_Status"] == "source_blank" for r in rows) == 6
    assert sum(r["Review_Status"] in {"ambiguous", "illegible"} for r in rows) == 0
    assert sum(response_count(r) for r in rows) == 607
    assert sum(response_count(r) for r in targets) == 504
    assert sum(response_count(r) for r in control_rows) == 103
    assert len(guard.stage_target_forms(rows)) == 504


def test_page_and_column_continuations_are_explicit():
    rows = guard.load_manual_cells(LEDGER)
    by_key = {(r["Item"], r["Site_Code"]): r for r in rows}
    expected = {
        ("165", "BMU"): ("67", "right"),
        ("165", "DBM"): ("68", "left"),
        ("170", "NAS"): ("68", "right"),
        ("170", "NGO"): ("69", "left"),
        ("174", "NAS"): ("69", "right"),
        ("174", "NGO"): ("70", "left"),
        ("183", "GUJ"): ("71", "right"),
        ("183", "MAR"): ("72", "left"),
        ("187", "NAS"): ("72", "right"),
        ("187", "NGO"): ("73", "left"),
        ("189", "TKO"): ("73", "right"),
        ("189", "NJA"): ("74", "left"),
    }
    for key, coordinate in expected.items():
        assert (by_key[key]["PDF_Page"], by_key[key]["Column"]) == coordinate
        assert int(by_key[key]["Printed_Page"]) == int(by_key[key]["PDF_Page"]) - 6


def test_difficult_900_dpi_readings_are_diplomatic():
    rows = guard.load_manual_cells(LEDGER)
    by_key = {(r["Item"], r["Site_Code"]): r for r in rows}
    expected = {
        ("165", "DBA"): "kɔrɔ",
        ("170", "NGO"): "kɛhinʌhɔi",
        ("170", "NTO"): "kolakh-dzat̪iɳ",
        ("170", "MAR"): "konʈiɖ-prʌkʌrtse",
        ("178", "BMU"): "puʈiio",
        ("179", "DBA"): "t̪huɽɔ",
        ("183", "BMU"): "sauwio | sau",
        ("184", "NAS"): "pukh̪lagi",
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
        writer.writerows([{**row, "OCR_Evidence": "never accepted"} for row in rows])
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


def test_cumulative_chunks_are_disjoint_and_match_manifest():
    ledgers = [HERE / f"items_{start:03d}_{end:03d}_hand_keyed.tsv" for start, end in [
        (1, 27), (28, 54), (55, 81), (82, 108), (109, 135), (136, 162), (163, 189)
    ]]
    rows = [row for ledger in ledgers for row in guard.load_manual_cells(ledger)]
    assert len(rows) == 189 * 17 == 3213
    assert len({(r["Item"], r["Site_Code"]) for r in rows}) == 3213
    assert {int(r["Item"]) for r in rows} == set(range(1, 190))
    assert sum(r["Review_Status"] == "attested" for r in rows) == 3169
    assert sum(r["Review_Status"] == "source_blank" for r in rows) == 44
    assert sum(response_count(r) for r in rows) == 3896
    assert len(guard.stage_target_forms(rows)) == 3130

    review = json.loads((PACKAGE / "source_manifest.json").read_text(encoding="utf-8"))["manual_review"]
    assert review["reviewed_cells"] == 3570
    assert review["attested_cells"] == 3526
    assert review["source_blank_cells"] == 44
    assert review["expanded_responses"] == 4385
    assert review["installed_forms"] == 2714
    assert review["remaining_cells"] == 0
