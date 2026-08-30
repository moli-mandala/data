"""Regression tests for the OCR-heavy SIL ESR 2018-010 Irula ingest."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_irula_2018"
IMPORTER = SOURCE_DIR / "import_irula.py"
INSTALLED = ROOT / "data/other/forms/20260828-sil-nilgiri-irula.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-nilgiri-irula-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-nilgiri-irula-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
SOURCE_KEY = "ernest-oleary-kelsall2018irula"
TARGET_CODES = {"KUN", "KOL", "CHE", "KIL", "MET", "CHO", "MAV", "ANA", "BOO", "THA", "NEL"}
CONTROL_CODES = {"CBT", "MAD", "KAN", "BAD", "ALU", "BET", "JEN"}


def forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def audited():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_importer_rebuilds_checked_artifacts_without_source_pdf():
    result = subprocess.run(
        [sys.executable, str(IMPORTER)], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert "installed=2054" in result.stdout
    assert "target_gaps=15" in result.stdout
    assert "controls=1319" in result.stdout
    assert "audit=3417" in result.stdout
    assert "unparsed=0" in result.stdout


def test_installed_scope_keys_and_transcription_contract():
    rows = forms()
    assert len(rows) == 2054
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert {row["Language_ID"] for row in rows} == {"Irula"}
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(row["Native"] == row["Parameter_ID"] == "" for row in rows)
    assert all(
        row[field] == ""
        for row in rows
        for field in (
            "Cognateset", "Etymology", "Variant_Of_Key", "Borrowed_From_Key",
            "Derivation_Parent_Keys",
        )
    )
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix B, printed p. ") for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)


def test_complete_audit_reconciles_source_responses_and_layout():
    audit = audited()
    assert len(audit) == 3417
    assert Counter(row["Status"] for row in audit) == Counter(installed=2054, excluded=1363)
    assert Counter(row["Record_Type"] for row in audit) == Counter(
        {"wordlist response": 3388, "layout fragment": 29}
    )
    wordlists = [row for row in audit if row["Record_Type"] == "wordlist response"]
    assert {int(row["Gloss_Number"]) for row in wordlists} == set(range(1, 188))
    assert {row["Site_Code"] for row in wordlists} == TARGET_CODES | CONTROL_CODES
    assert len([row for row in wordlists if row["Site_Code"] in CONTROL_CODES]) == 1319
    assert all(
        row["Status"] == "excluded"
        for row in wordlists if row["Site_Code"] in CONTROL_CODES
    )
    gaps = [row for row in wordlists if row["Review"] == "missing"]
    assert len(gaps) == 15 and all(row["Status"] == "excluded" for row in gaps)
    assert not [row for row in audit if row["Status"] in {"unparsed", "unmapped"}]
    mad = [row for row in wordlists if row["Site_Code"] == "MAD"]
    assert mad and all("does not expand the code MAD" in row["Reason"] for row in mad)


def test_review_and_uncertainty_counts_are_pinned():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_pdf_sha256"] == "2e5a4ef0f4c941437d09a1c8fa49ba01d4fe79e0915ad9248ac7b83280fb4c62"
    assert manifest["counts"]["target_response_records"] == 2069
    assert manifest["counts"]["installed_target_forms"] == 2054
    assert manifest["counts"]["target_source_gaps"] == 15
    assert manifest["counts"]["uncertainty_flags"] == {
        "source-raster-affricate": 120,
        "source-raster-coronal": 1001,
        "source-raster-labial": 9,
        "source-raster-legacy-glyph-ɪ": 100,
        "source-raster-length": 1,
        "source-raster-nasalization": 17,
        "source-raster-unreadable-segment": 1,
        "source-raster-vowel": 52,
    }


def test_representative_source_entries_and_multiple_responses():
    by_key = {row["Entry_Key"]: row for row in forms()}
    assert by_key["silirula2018:g001:nilgiri-irula-kunjapanai:i1"]["Form"] == "oɖʌmbɨ"
    assert by_key["silirula2018:g004:nilgiri-irula-thaliyur:i1"]["Form"] == "muɲʤi"
    assert by_key["silirula2018:g183:nilgiri-irula-bookapuram:i2"]["Form"] == "ʌʋã"
    assert "source-raster-nasalization" in by_key[
        "silirula2018:g183:nilgiri-irula-bookapuram:i2"
    ]["Notes"]


def test_source_and_all_eleven_dialects_are_registered():
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    ids = {row["Tags"].split(":")[2] for row in forms()}
    assert len(ids) == 11
    for dialect_id in ids:
        assert dialects[dialect_id]["Language_ID"] == "Irula"
        assert dialects[dialect_id]["Glottocode"] == "irul1243"
        assert dialects[dialect_id]["Quality"] == "C"
        assert "prints no village coordinates" in dialects[dialect_id]["Location"]


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {part.split("[", 1)[0].strip() for part in row["Source"].split(";")}
        ]
    if not compiled:
        pytest.skip("cldf/forms.csv predates the Irula ingest; enforced after make all")
    assert len(compiled) == len(forms())
    assert {row["Language_ID"] for row in compiled} == {"Irula"}
    # ``make_cldf.py`` emits the graph-neutral rows before ``unify_cldf.py``
    # adds Status.  Accept either build stage while enforcing the same state.
    if "Status" in compiled[0]:
        assert all(row["Status"] == "unlinked" for row in compiled)
    else:
        assert all(row["Parameter_ID"] == row["Cognateset"] == "" for row in compiled)
    assert all(row["Original"] for row in compiled)
