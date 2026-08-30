import csv
import importlib.util
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mavilan_import", ROOT / "import_mavilan_tulu_2025.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MOD)


def test_canonical_source_registry_and_corrected_topology():
    assert MOD.PDF.exists()
    assert MOD.sha256(MOD.PDF) == MOD.PDF_SHA256
    assert MOD.EXPECTED_ITEMS == 208
    assert MOD.EXPECTED_CELLS == 1248
    registry = MOD.load_registry()
    assert set(registry) == set(MOD.SITE_ORDER)
    assert registry["KOD"]["Printed_Final_Page_Code"] == "IKOD"
    assert [registry[site]["Language_ID"] for site in ("MTP", "MTV", "MTE")] == [
        "markodi_pannithadam", "markodi_vannarkadav", "markodi_ennappara"
    ]


def test_complete_manual_ledger_is_disjoint_ocr_blind_nfc_and_resolved():
    rows = MOD.load_cells()
    expected_keys = {
        (item, site) for item in range(1, 209) for site in MOD.SITE_ORDER
    }
    assert len(rows) == 1248
    assert {(int(row["Item"]), row["Site_Code"]) for row in rows} == expected_keys
    assert not any(int(row["Item"]) in {209, 210} for row in rows)
    assert Counter(row["Review_Status"] for row in rows) == Counter(
        attested=1230, source_blank=18
    )
    assert all(row["Reviewer_Declaration"] == MOD.DECLARATION for row in rows)
    assert all(row["Reviewer_Method"] == MOD.METHOD for row in rows)
    assert all(unicodedata.is_normalized("NFC", value) for row in rows for value in row.values())
    assert not any("ocr" in key.casefold() for key in rows[0])
    assert MOD.summarize(rows) == {
        "reviewed_cells": 1248,
        "attested_cells": 1230,
        "source_blank_cells": 18,
        "ambiguous_cells": 0,
        "illegible_cells": 0,
        "target_reviewed_cells": 624,
        "control_reviewed_cells": 624,
        "pending_cells": 0,
    }


def test_staging_has_exact_target_control_and_blank_accounting():
    forms, audit = MOD.stage(MOD.load_cells())
    assert len(forms) == 615
    assert len(audit) == 1248
    assert Counter(row["Disposition"] for row in audit) == Counter({
        "target-form": 615,
        "control-excluded": 615,
        "source-blank": 18,
    })
    assert Counter(row["Site_Code"] in MOD.TARGETS for row in audit if row["Review_Status"] == "source_blank") == Counter({True: 9, False: 9})
    expected_fields = [
        "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
        "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
        "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
    ]
    assert list(forms[0]) == expected_fields
    assert len({row["Entry_Key"] for row in forms}) == 615
    assert all(not row["Parameter_ID"] for row in forms)
    assert all(row["Form"] == row["Phonemic"] for row in forms)
    assert all(row["Source"].startswith("canvin2025[p. ") for row in forms)
    assert {row["Language_ID"] for row in forms} == {
        "markodi_pannithadam", "markodi_vannarkadav", "markodi_ennappara"
    }


def test_items_190_208_have_exact_direct_source_page_topology():
    rows = [row for row in MOD.load_cells() if 190 <= int(row["Item"]) <= 208]
    assert len(rows) == 114
    assert all(row["Review_Status"] == "attested" for row in rows)
    assert all(row["Confidence"] == "high" for row in rows)
    expected = {}
    for item in range(190, 194):
        expected[item] = ("37", "31", "left", str(item - 186))
    for item in range(194, 201):
        expected[item] = ("37", "31", "middle", str(item - 193))
    for item in range(201, 208):
        expected[item] = ("37", "31", "right", str(item - 200))
    expected[208] = ("38", "32", "left", "1")
    assert {
        (int(row["Item"]), row["PDF_Page"], row["Printed_Page"], row["Column"], row["Page_Row"])
        for row in rows
    } == {(item, *coordinate) for item, coordinate in expected.items()}


def test_explicit_source_blanks_and_unresolved_ledger_are_exact():
    blanks = [row for row in MOD.load_cells() if row["Review_Status"] == "source_blank"]
    assert [(row["Item"], row["Site_Code"], row["Uncertainty"]) for row in blanks] == [
        ("29", "KOD", "source prints NA"),
        ("30", "KOD", "source prints Nill"),
        ("41", "KOD", "source prints Nill"),
        ("48", "TUL", "source prints Nill"),
        ("67", "MTP", "source prints Nill"),
        ("67", "MTV", "source prints Nill"),
        ("67", "MTE", "source prints Nill"),
        ("67", "MAL", "source prints Nill"),
        ("67", "TUL", "source prints Nill"),
        ("70", "MTP", "source prints Nill"),
        ("70", "MTV", "source prints Nill"),
        ("70", "MTE", "source prints Nill"),
        ("70", "TUL", "source prints Nill"),
        ("76", "MTP", "source prints Nill"),
        ("76", "MTV", "source prints Nill"),
        ("76", "MTE", "source prints Nill"),
        ("76", "TUL", "source prints Nill"),
        ("157", "KOD", "source prints Nill"),
    ]
    with (ROOT / "unresolved_readings.tsv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle, delimiter="\t")) == []


def test_existing_profile_covers_every_staged_form_without_additions():
    forms, _ = MOD.stage(MOD.load_cells())
    inventory, additions = MOD.profile_inventory(forms)
    assert len(inventory) == 59
    assert all(row["Covered_By_Existing_Profile"] == "yes" for row in inventory)
    assert additions == []


def test_post_entry_legacy_reconciliation_is_complete_and_audited():
    rows = MOD.load_cells()
    manual = {
        (row["Gloss"], row["Site_Code"]): row["Manual_Transcription"]
        for row in rows
        if row["Site_Code"] in MOD.TARGETS and row["Review_Status"] == "attested"
    }
    language_to_site = {
        "markodi_pannithadam": "MTP",
        "markodi_vannarkadav": "MTV",
        "markodi_ennappara": "MTE",
    }
    legacy_path = ROOT.parents[1] / "20260723-markodi.csv"
    with legacy_path.open(encoding="utf-8", newline="") as handle:
        legacy = {
            (row[3], language_to_site[row[0]]): row[2]
            for row in csv.reader(handle)
        }
    assert set(manual) == set(legacy)
    exact = sum(manual[key] == legacy[key] for key in manual)
    assert (exact, len(manual) - exact) == (556, 59)
