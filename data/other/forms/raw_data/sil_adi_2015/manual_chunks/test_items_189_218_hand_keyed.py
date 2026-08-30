import importlib.util
import unicodedata
from collections import Counter
from pathlib import Path


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_189_218_hand_keyed.tsv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("adi_guard_items_189_218", PACKAGE / "import_adi_2015.py")


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
    assert len(guard.stage_forms(rows)) == 272
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert (by_key[("192", "BR")]["PDF_Page"], by_key[("192", "BR")]["Column"]) == ("30", "left")
    assert (by_key[("192", "RM")]["PDF_Page"], by_key[("192", "RM")]["Column"]) == ("30", "middle")
    assert (by_key[("202", "BR")]["PDF_Page"], by_key[("202", "BR")]["Column"]) == ("30", "right")
    assert (by_key[("202", "RM")]["PDF_Page"], by_key[("202", "RM")]["Column"]) == ("31", "left")
    assert (by_key[("207", "MN")]["PDF_Page"], by_key[("207", "MN")]["Column"]) == ("31", "left")
    assert (by_key[("207", "BR")]["PDF_Page"], by_key[("207", "BR")]["Column"]) == ("31", "middle")
    assert (by_key[("217", "MN")]["PDF_Page"], by_key[("217", "MN")]["Column"]) == ("32", "left")
    assert by_key[("203", "PD")]["Manual_Transcription"] == "ərpaknam | ərpaknam"
    assert by_key[("210", "BR")]["Source_Cognate_Labels"] == "1 | 2"
    assert by_key[("218", "PD")]["Manual_Transcription"] == "jad̪nam"


def test_cumulative_guard_and_incomplete_stage_refusal():
    ledgers = [path for path in sorted(HERE.glob("items_*_hand_keyed.tsv"))
               if int(path.name.split("_")[1]) <= 189]
    rows = guard.load_manual_ledgers(ledgers)
    assert Counter(row["Review_Status"] for row in rows) == Counter(attested=1870, source_blank=92)
    assert len(rows) == 1962 and len(guard.stage_forms(rows)) == 1942
    try:
        guard.require_full_review(rows)
    except RuntimeError as error:
        assert str(error) == "manual visual review incomplete: 801 of 2763 cells unreviewed"
    else:
        raise AssertionError("partial review must not stage")
