import importlib.util
import unicodedata
from collections import Counter
from pathlib import Path


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_249_278_hand_keyed.tsv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("adi_guard_items_249_278", PACKAGE / "import_adi_2015.py")


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
    assert Counter(row["Review_Status"] for row in rows) == Counter(
        attested=269, source_blank=1
    )
    assert len(guard.stage_forms(rows)) == 282
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert (by_key[("251", "BR")]["PDF_Page"], by_key[("251", "BR")]["Column"]) == ("34", "left")
    assert (by_key[("251", "RM")]["PDF_Page"], by_key[("251", "RM")]["Column"]) == ("34", "middle")
    assert (by_key[("261", "MN")]["PDF_Page"], by_key[("261", "MN")]["Column"]) == ("34", "right")
    assert (by_key[("261", "BR")]["PDF_Page"], by_key[("261", "BR")]["Column"]) == ("35", "left")
    assert (by_key[("265", "PL")]["PDF_Page"], by_key[("265", "PL")]["Column"]) == ("35", "left")
    assert (by_key[("265", "AS")]["PDF_Page"], by_key[("265", "AS")]["Column"]) == ("35", "middle")
    assert (by_key[("270", "ML")]["PDF_Page"], by_key[("270", "ML")]["Column"]) == ("35", "middle")
    assert (by_key[("270", "PL")]["PDF_Page"], by_key[("270", "PL")]["Column"]) == ("35", "right")
    assert (by_key[("275", "BR")]["PDF_Page"], by_key[("275", "BR")]["Column"]) == ("35", "right")
    assert (by_key[("275", "RM")]["PDF_Page"], by_key[("275", "RM")]["Column"]) == ("36", "left")
    blank = by_key[("277", "RM")]
    assert blank["Review_Status"] == "source_blank"
    assert blank["Manual_Transcription"] == ""
    assert blank["Source_Cognate_Labels"] == "0"
    assert blank["Uncertainty"] == "Source prints cognate label 0 and ‘no entry’."
    assert by_key[("270", "BK")]["Manual_Transcription"] == "put̪urna | ʃe nnə"
    assert by_key[("275", "MN")]["Source_Cognate_Labels"] == "1 | 4"


def test_cumulative_guard_and_incomplete_stage_refusal():
    paths = [path for path in sorted(HERE.glob("items_*_hand_keyed.tsv"))
             if int(path.name.split("_")[1]) <= 249]
    rows = guard.load_manual_ledgers(paths)
    assert Counter(row["Review_Status"] for row in rows) == Counter(
        attested=2409, source_blank=93
    )
    assert len(rows) == 2502 and len(guard.stage_forms(rows)) == 2501
    try:
        guard.require_full_review(rows)
    except RuntimeError as error:
        assert str(error) == "manual visual review incomplete: 261 of 2763 cells unreviewed"
    else:
        raise AssertionError("partial review must not stage")
