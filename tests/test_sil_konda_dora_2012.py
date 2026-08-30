"""Focused checks for the image-only ESR 2012-016 Konda Dora ingest."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_konda_dora_2012"
IMPORTER = SOURCE_DIR / "import_konda_dora.py"
REVIEWED = SOURCE_DIR / "reviewed_transcription.psv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-konda-dora.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-konda-dora-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-konda-dora-manifest.json"
PROFILE = ROOT / "conversion/sil-konda-dora.txt"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]


def reviewed():
    with REVIEWED.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="|"))


def forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def audited():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_source_local_importer_rebuilds_checked_artifacts():
    result = subprocess.run(
        [sys.executable, str(IMPORTER), "--install"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert "conceptual_source_cells_manually_reviewed=856" in result.stdout
    assert "confirmed_blank_cells=129" in result.stdout
    assert "installed_forms_after_source_defined_expansion=452" in result.stdout
    assert "unresolved_or_illegible_cells=0" in result.stdout


def test_manifest_pins_source_and_complete_review_denominator():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["pdf_sha256"] == "6e0a3e5522a45752938f8279753d07b4e29d7b76ca73e88f71c4e283dfd0f533"
    assert manifest["pdf_pages"] == 106
    assert manifest["counts"] == {
        "prompts": 214,
        "lists": 4,
        "conceptual_source_cells_manually_reviewed": 856,
        "target_cells_manually_reviewed": 428,
        "control_cells_manually_reviewed": 428,
        "attested_cells": 727,
        "confirmed_blank_cells": 129,
        "confirmed_blank_target_cells": 43,
        "confirmed_blank_control_cells": 86,
        "excluded_control_response_cells": 342,
        "installed_forms_after_source_defined_expansion": 452,
        "audit_rows": 856,
        "unresolved_or_illegible_cells": 0,
        "source_marked_uncertain_cells": 0,
    }
    assert manifest["review"]["unresolved"] == []
    assert "never supplies an accepted reading" in manifest["review"]["ocr"]


def test_every_cell_is_manually_reviewed_and_page_located():
    rows = reviewed()
    assert len(rows) == 214
    assert len({row["Prompt_Key"] for row in rows}) == 214
    assert {row["Review_Status"] for row in rows} == {"manually_verified"}
    assert {row["Confidence"] for row in rows} == {"high"}
    assert {int(row["Target_PDF_Page"]) for row in rows} == set(range(89, 98))
    assert {int(row["Control_PDF_Page"]) for row in rows} == set(range(98, 107))
    assert sum(1 for row in rows for list_name in ("Koraput", "Visakh", "Telugu", "Adivasi_Oriya")) == 856
    assert {(row["Prompt_Key"], row["Gloss"]) for row in rows if row["Item"] == "212"} == {
        ("212-liver", "liver"), ("212-foot", "foot"),
    }


def test_audit_accounts_for_targets_controls_blanks_and_expansions():
    rows = audited()
    assert len(rows) == 856
    assert Counter(row["Role"] for row in rows) == Counter({"target": 428, "comparison control": 428})
    assert Counter(row["Record_Type"] for row in rows) == Counter({"response": 727, "blank": 129})
    assert Counter(row["Status"] for row in rows) == Counter({"excluded": 471, "installed": 385})
    assert sum(int(row["Installed_Count"]) for row in rows) == 452
    assert all("manual cell-by-cell" in row["Review_Method"] for row in rows)
    assert all("OCR/text layer used only" in row["Review_Method"] for row in rows)
    assert not [row for row in rows if row["Confidence"] != "high"]
    assert not [row for row in rows if row["Review_Status"] != "manually_verified"]


def test_installed_rows_are_diplomatic_stable_and_target_only():
    rows = forms()
    assert len(rows) == 452
    assert Counter(row["Tags"].split(":")[2] for row in rows) == Counter({
        "sil-konda-dora-1987-koraput": 231,
        "sil-konda-dora-1987-visakh": 221,
    })
    assert {row["Language_ID"] for row in rows} == {"Konda"}
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(row["Parameter_ID"] == row["Native"] == "" for row in rows)
    assert all(row["Source"].startswith("blair-george2012kondadora[Appendix 9.5") for row in rows)
    assert all(row["Cognateset"] == row["Etymology"] == "" for row in rows)
    assert all(row["Variant_Of_Key"] == row["Borrowed_From_Key"] == row["Derivation_Parent_Keys"] == "" for row in rows)


def test_representative_manual_cells_and_source_defined_splits_survive():
    rows = audited()
    by_cell = {(row["Prompt_Key"], row["List"]): row for row in rows}
    expected = {
        ("001", "Koraput"): ("1oɽol", "oɽol", "1"),
        ("074", "Koraput"): ("-bumi ka:ṭoliŋ", "bumi ka:ṭoliŋ", "ungrouped"),
        ("122", "Visakh"): ("1i?en", "i?en", "1"),
        ("187", "Koraput"): ("1sundziṭa:n/sudz?a/----", "sundziṭa:n/sudz?a/----", "1"),
        ("195", "Visakh"): ("2----/----/naḍiḍeŋ", "----/----/naḍiḍeŋ", "2"),
        ("212-foot", "Koraput"): ("1pa:ḍam", "pa:ḍam", "1"),
    }
    for key, (source_cell, manual, group) in expected.items():
        row = by_cell[key]
        assert (row["Source_Cell"], row["Manual_Form"], row["Similarity_Group"]) == (source_cell, manual, group)
    assert by_cell[("187", "Koraput")]["Expanded_Forms"] == "sundziṭa:n | sudz?a"
    assert by_cell[("195", "Visakh")]["Expanded_Forms"] == "naḍiḍeŋ"


def test_source_profile_covers_all_installed_forms_and_maps_source_conventions():
    tokenizer = Tokenizer(str(PROFILE))
    for row in forms():
        converted = tokenizer(
            unicodedata.normalize("NFC", row["Form"]), column="IPA",
            segment_separator="", separator="",
        )
        assert "�" not in converted
    assert tokenizer("dza:va", column="IPA", segment_separator="", separator="") == "jāva".replace("ā", "aː")
    assert tokenizer("uṇ?a", column="IPA", segment_separator="", separator="") == "uṇʔa"
    assert tokenizer("oɽol", column="IPA", segment_separator="", separator="") == "oṛol"
