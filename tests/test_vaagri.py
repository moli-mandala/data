import importlib.util
import csv
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/vaagri.py"
SPEC = importlib.util.spec_from_file_location("vaagri_extractor", SCRIPT)
vaagri = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = vaagri
SPEC.loader.exec_module(vaagri)


def test_numbering_includes_author_insertion():
    items = vaagri.expected_items()
    assert len(items) == 2441
    assert "526a" in items
    assert items[1800:1804] == ["1798", "1798a", "1799", "1800"]
    assert "1897a" in items
    assert items[-1] == "2436"


def test_entry_parser_separates_source_comparison():
    entry = vaagri.split_entry(
        "12", "inda:ru, n., neut. - darkness (C. andha:r >386)", 186
    )
    assert entry.form == "inda:ru"
    assert entry.gloss == "darkness"
    assert entry.etymology == "(C. andha:r >386)"
    assert entry.printed_page == 172
    assert vaagri.grammatical_tags(entry.morphology) == ["noun", "n"]


def test_import_row_has_stable_key_and_page_locator():
    entry = vaagri.Entry("1", 186, 172, "i", "adj.", "this", "", "i, adj. - this")
    row = vaagri.import_rows([entry], [])[0]
    assert len(row) == 15
    assert row[0:4] == ["VB", "", "i", "this"]
    assert row[7] == "srinivasa[p. 186 (printed p. 172), item 1]"
    assert row[10] == "srinivasa:1"
    assert row[14] == "adj"


def test_generated_vaagri_import_is_complete_and_traceable():
    forms_path = SCRIPT.parents[1] / "20220913-vaagri.csv"
    raw_path = SCRIPT.parent / "vaagri_dictionary.csv"
    with forms_path.open(encoding="utf-8", newline="") as stream:
        forms = list(csv.reader(stream))
    with raw_path.open(encoding="utf-8", newline="") as stream:
        raw = list(csv.DictReader(stream))

    assert len(raw) == 2441
    assert len(forms) == 2456  # trusted slash/alternate heads are separate forms
    assert all(len(row) == 15 and row[0] == "VB" and row[2] for row in forms)
    assert all(row[7].startswith("srinivasa[p. ") for row in forms)
    generated_keys = [row[10] for row in forms if row[10]]
    assert len(generated_keys) == len(set(generated_keys)) == 2301
