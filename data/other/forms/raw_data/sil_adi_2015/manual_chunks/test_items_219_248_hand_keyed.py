import importlib.util
import unicodedata
from collections import Counter
from pathlib import Path


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_219_248_hand_keyed.tsv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("adi_guard_items_219_248", PACKAGE / "import_adi_2015.py")


def test_complete_visually_reviewed_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 30 * 9 == 270
    assert len({(row["Item"], row["Site_Code"]) for row in rows}) == 270
    assert {row["Reviewer_Declaration"] for row in rows} == {guard.DECLARATION}
    assert all("OCR" not in key.upper() for key in rows[0])
    assert all(unicodedata.is_normalized("NFC", value)
               for row in rows for value in row.values())


def test_accounting_coordinates_and_form_expansion():
    rows = guard.load_manual_cells(LEDGER)
    assert Counter(row["Review_Status"] for row in rows) == Counter(attested=270)
    assert len(guard.stage_forms(rows)) == 277
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert (by_key[("226", "PL")]["PDF_Page"], by_key[("226", "PL")]["Column"]) == ("32", "middle")
    assert (by_key[("226", "AS")]["PDF_Page"], by_key[("226", "AS")]["Column"]) == ("32", "right")
    assert (by_key[("231", "ML")]["PDF_Page"], by_key[("231", "ML")]["Column"]) == ("32", "right")
    assert (by_key[("231", "PL")]["PDF_Page"], by_key[("231", "PL")]["Column"]) == ("33", "left")
    assert (by_key[("236", "RM")]["PDF_Page"], by_key[("236", "RM")]["Column"]) == ("33", "left")
    assert (by_key[("236", "ML")]["PDF_Page"], by_key[("236", "ML")]["Column"]) == ("33", "middle")
    assert (by_key[("246", "BR")]["PDF_Page"], by_key[("246", "BR")]["Column"]) == ("33", "right")
    assert (by_key[("246", "RM")]["PDF_Page"], by_key[("246", "RM")]["Column"]) == ("34", "left")
    assert by_key[("222", "BR")]["Manual_Transcription"] == "ʃinam | ʃinam"
    assert by_key[("231", "PL")]["Source_Cognate_Labels"] == "1 | 1"
    assert by_key[("244", "BK")]["Manual_Transcription"] == "t̪ənə̃"


def test_cumulative_guard_and_incomplete_stage_refusal():
    paths = [path for path in sorted(HERE.glob("items_*_hand_keyed.tsv"))
             if int(path.name.split("_")[1]) <= 219]
    rows = guard.load_manual_ledgers(paths)
    assert Counter(row["Review_Status"] for row in rows) == Counter(attested=2140, source_blank=92)
    assert len(rows) == 2232 and len(guard.stage_forms(rows)) == 2219
    try:
        guard.require_full_review(rows)
    except RuntimeError as error:
        assert str(error) == "manual visual review incomplete: 531 of 2763 cells unreviewed"
    else:
        raise AssertionError("partial review must not stage")
