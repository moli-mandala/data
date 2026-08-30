import csv
import importlib.util
import unicodedata
from collections import Counter
from pathlib import Path


HERE = Path(__file__).parent
PACKAGE = HERE.parent
LEDGER = HERE / "items_279_307_hand_keyed.tsv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


guard = load_module("adi_guard_items_279_307", PACKAGE / "import_adi_2015.py")


def test_complete_visually_reviewed_chunk():
    rows = guard.load_manual_cells(LEDGER)
    assert len(rows) == 29 * 9 == 261
    assert len({(row["Item"], row["Site_Code"]) for row in rows}) == 261
    assert Counter(row["Review_Status"] for row in rows) == Counter(attested=261)
    assert {row["Reviewer_Declaration"] for row in rows} == {guard.DECLARATION}
    assert {row["Reviewer_Method"] for row in rows} == {guard.METHOD}
    assert all("OCR" not in key.upper() for key in rows[0])
    assert all(unicodedata.is_normalized("NFC", value)
               for row in rows for value in row.values())
    assert len(guard.stage_forms(rows)) == 269


def test_coordinates_punctuation_and_response_cardinality():
    rows = guard.load_manual_cells(LEDGER)
    by_key = {(row["Item"], row["Site_Code"]): row for row in rows}
    assert (by_key[("284", "SM")]["PDF_Page"], by_key[("284", "SM")]["Column"]) == ("36", "middle")
    assert (by_key[("284", "BK")]["PDF_Page"], by_key[("284", "BK")]["Column"]) == ("36", "right")
    assert (by_key[("289", "AS")]["PDF_Page"], by_key[("289", "AS")]["Column"]) == ("36", "right")
    assert (by_key[("289", "PD")]["PDF_Page"], by_key[("289", "PD")]["Column"]) == ("37", "left")
    assert (by_key[("294", "AS")]["PDF_Page"], by_key[("294", "AS")]["Column"]) == ("37", "left")
    assert (by_key[("294", "PD")]["PDF_Page"], by_key[("294", "PD")]["Column"]) == ("37", "middle")
    assert (by_key[("304", "ML")]["PDF_Page"], by_key[("304", "ML")]["Column"]) == ("37", "right")
    assert (by_key[("304", "PL")]["PDF_Page"], by_key[("304", "PL")]["Column"]) == ("38", "left")
    assert (by_key[("306", "SM")]["PDF_Page"], by_key[("306", "SM")]["Column"]) == ("38", "middle")
    assert (by_key[("306", "BK")]["PDF_Page"], by_key[("306", "BK")]["Column"]) == ("38", "right")
    assert by_key[("292", "RM")]["Manual_Transcription"] == "huupe?"
    assert by_key[("293", "AS")]["Manual_Transcription"] == "kapə?"
    assert by_key[("302", "BK")]["Manual_Transcription"] == "meju, mu"
    assert by_key[("303", "BK")]["Manual_Transcription"] == "muu | me j"
    assert by_key[("303", "BK")]["Source_Cognate_Labels"] == "1 | 4"


def test_complete_cumulative_package_and_staging():
    rows = guard.load_manual_ledgers(sorted(HERE.glob("items_*_hand_keyed.tsv")))
    assert len(rows) == 2763
    assert Counter(row["Review_Status"] for row in rows) == Counter(
        attested=2670, source_blank=93
    )
    guard.require_full_review(rows)
    assert len(guard.stage_forms(rows)) == 2770
    registry = guard.load_registry(PACKAGE / "list_registry.tsv")
    forms, audit = guard.build_source_package(rows, registry)
    assert len(forms) == 2770 and len(audit) == 2763
    assert Counter(row["Disposition"] for row in audit) == Counter(
        staged=2670, **{"blank-excluded": 93}
    )
    assert len({row["Entry_Key"] for row in forms}) == 2770
    assert {row["Language_ID"] for row in forms} == {
        "MisingPadamMiriMinyong", "BoriKarko", "BokarRamo", "Milang"
    }
    assert len({row["Tags"] for row in forms}) == 9
    assert all(row["Parameter_ID"] == row["Cognateset"] == "" for row in forms)
    assert all(unicodedata.is_normalized("NFC", value)
               for row in forms + audit for value in row.values())


def test_written_source_local_package_matches_builder():
    rows = guard.load_manual_ledgers(sorted(HERE.glob("items_*_hand_keyed.tsv")))
    expected_forms, expected_audit = guard.build_source_package(
        rows, guard.load_registry(PACKAGE / "list_registry.tsv")
    )
    with (PACKAGE / "staged_forms.csv").open(encoding="utf-8", newline="") as handle:
        actual_forms = list(csv.DictReader(handle, fieldnames=guard.RAW_FORM_FIELDS))
    with (PACKAGE / "staged_audit.tsv").open(encoding="utf-8", newline="") as handle:
        actual_audit = list(csv.DictReader(handle, delimiter="\t"))
    with (PACKAGE / "unresolved_readings.tsv").open(encoding="utf-8", newline="") as handle:
        unresolved = list(csv.DictReader(handle, delimiter="\t"))
    assert actual_forms == expected_forms
    assert actual_audit == expected_audit
    assert unresolved == []
    inventory = list(csv.DictReader(
        (PACKAGE / "symbol_inventory.tsv").open(encoding="utf-8"), delimiter="\t"
    ))
    assert inventory
    assert sum(int(row["Count"]) for row in inventory) == sum(
        len(row["Form"]) for row in expected_forms
    )
