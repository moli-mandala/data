import importlib.util
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_071_084_hand_keyed.tsv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("adi_guard_items_71_84", PACKAGE / "import_adi_2015.py")


def test_complete_visually_reviewed_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 14 * 9 == 126
    assert len({(row["Item"], row["Site_Code"]) for row in rows}) == 126
    assert {row["Reviewer_Declaration"] for row in rows} == {guard.DECLARATION}
    assert all("OCR" not in key.upper() for key in rows[0])
    assert all(unicodedata.is_normalized("NFC", value)
               for row in rows for value in row.values())


def test_accounting_and_form_expansion():
    rows = guard.load_manual_cells(LEDGER)
    assert sum(row["Review_Status"] == "attested" for row in rows) == 116
    assert sum(row["Review_Status"] == "source_blank" for row in rows) == 10
    assert sum(row["Review_Status"] in {"ambiguous", "illegible"}
               for row in rows) == 0
    assert len(guard.stage_forms(rows)) == 121
