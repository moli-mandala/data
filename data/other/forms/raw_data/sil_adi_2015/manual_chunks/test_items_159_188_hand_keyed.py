import importlib.util
import unicodedata
from collections import Counter
from pathlib import Path


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_159_188_hand_keyed.tsv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("adi_guard_items_159_188", PACKAGE / "import_adi_2015.py")


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
    assert Counter(row["Review_Status"] for row in rows) == Counter(attested=260, source_blank=10)
    assert len(guard.stage_forms(rows)) == 266
    blanks = {(row["Item"], row["Site_Code"]) for row in rows if row["Review_Status"] == "source_blank"}
    sites = {"MN", "BR", "RM", "ML", "PL", "AS", "PD", "SM", "BK"}
    assert blanks == {("160", "MN")} | {("168", code) for code in sites}
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert (by_key[("162", "PL")]["PDF_Page"], by_key[("162", "PL")]["Column"]) == ("28", "left")
    assert (by_key[("162", "AS")]["PDF_Page"], by_key[("162", "AS")]["Column"]) == ("28", "middle")
    assert (by_key[("172", "BR")]["PDF_Page"], by_key[("172", "BR")]["Column"]) == ("28", "right")
    assert (by_key[("172", "RM")]["PDF_Page"], by_key[("172", "RM")]["Column"]) == ("29", "left")
    assert (by_key[("187", "BR")]["PDF_Page"], by_key[("187", "BR")]["Column"]) == ("29", "right")
    assert (by_key[("187", "RM")]["PDF_Page"], by_key[("187", "RM")]["Column"]) == ("30", "left")
    assert by_key[("159", "RM")]["Manual_Transcription"] == "joʔʃikh"
    assert by_key[("164", "MN")]["Manual_Transcription"] == "ɲud̪uuŋ | ɲud̪uuŋ"
    assert by_key[("166", "SM")]["Source_Cognate_Labels"] == "1 | 3"
    assert by_key[("188", "BK")]["Manual_Transcription"] == "d̪õnam"


def test_cumulative_guard_and_incomplete_stage_refusal():
    ledgers = [path for path in sorted(HERE.glob("items_*_hand_keyed.tsv"))
               if int(path.name.split("_")[1]) <= 159]
    rows = guard.load_manual_ledgers(ledgers)
    assert Counter(row["Review_Status"] for row in rows) == Counter(attested=1600, source_blank=92)
    assert len(rows) == 1692 and len(guard.stage_forms(rows)) == 1670
    try:
        guard.require_full_review(rows)
    except RuntimeError as error:
        assert str(error) == "manual visual review incomplete: 1071 of 2763 cells unreviewed"
    else:
        raise AssertionError("partial review must not stage")
