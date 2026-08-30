import importlib.util
import unicodedata
from collections import Counter
from pathlib import Path


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_129_158_hand_keyed.tsv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("adi_guard_items_129_158", PACKAGE / "import_adi_2015.py")


def test_complete_visually_reviewed_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 30 * 9 == 270
    assert len({(row["Item"], row["Site_Code"]) for row in rows}) == 270
    assert {row["Reviewer_Declaration"] for row in rows} == {guard.DECLARATION}
    assert all("OCR" not in key.upper() for key in rows[0])
    assert all(unicodedata.is_normalized("NFC", value)
               for row in rows for value in row.values())


def test_accounting_coordinates_blanks_and_form_expansion():
    rows = guard.load_manual_cells(LEDGER)
    assert Counter(row["Review_Status"] for row in rows) == Counter(attested=262, source_blank=8)
    assert len(guard.stage_forms(rows)) == 278
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    assert blanks == {
        ("145", "MN"), ("145", "PL"), ("145", "PD"), ("145", "SM"),
        ("147", "RM"), ("147", "SM"), ("150", "BR"), ("150", "SM"),
    }
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert (by_key[("129", "MN")]["PDF_Page"], by_key[("129", "MN")]["Column"]) == ("25", "right")
    assert (by_key[("129", "RM")]["PDF_Page"], by_key[("129", "RM")]["Column"]) == ("26", "left")
    assert (by_key[("152", "BK")]["PDF_Page"], by_key[("152", "BK")]["Column"]) == ("27", "right")
    assert (by_key[("157", "BK")]["PDF_Page"], by_key[("157", "BK")]["Column"]) == ("28", "left")
    assert by_key[("137", "PD")]["Manual_Transcription"] == "ami | ami | m mi"
    assert by_key[("158", "PD")]["Manual_Transcription"] == "kot̪up"


def test_cumulative_guard_and_incomplete_stage_refusal():
    ledgers = [path for path in sorted(HERE.glob("items_*_hand_keyed.tsv"))
               if int(path.name.split("_")[1]) <= 129]
    rows = guard.load_manual_ledgers(ledgers)
    assert Counter(row["Review_Status"] for row in rows) == Counter(attested=1340, source_blank=82)
    assert len(rows) == 1422 and len(guard.stage_forms(rows)) == 1404
    try:
        guard.require_full_review(rows)
    except RuntimeError as error:
        assert str(error) == "manual visual review incomplete: 1341 of 2763 cells unreviewed"
    else:
        raise AssertionError("partial review must not stage")
