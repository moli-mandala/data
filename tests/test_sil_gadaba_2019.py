"""Regression tests for the OCR-heavy SIL ESR 2019-005 Gadaba ingest."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_gadaba_2019"
IMPORTER = SOURCE_DIR / "import_gadaba.py"
INSTALLED = ROOT / "data/other/forms/20260828-sil-mudhili-gadaba.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-mudhili-gadaba-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-mudhili-gadaba-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
SOURCE_KEY = "adimathara2019mudhili"
DIALECT_IDS = {
    "sil-gadaba-2019-bobbilivalasa",
    "sil-gadaba-2019-gogaduvalasa",
    "sil-gadaba-2019-panukuvalasa",
    "sil-gadaba-2019-reyavanivalasa",
    "sil-gadaba-2019-kothavalasa",
    "sil-gadaba-2019-suregadivalasa",
    "sil-gadaba-2019-chinachipuruvalasa",
}


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
    assert "installed=1538" in result.stdout
    assert "target_gaps=8" in result.stdout
    assert "controls=214" in result.stdout
    assert "audit=1765" in result.stdout


def test_installed_scope_keys_and_transcription_contract():
    rows = forms()
    assert len(rows) == 1538
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert {row["Language_ID"] for row in rows} == {"Gadaba"}
    assert all(row["Form"] and row["Phonemic"] for row in rows)
    assert all(not any(mark in row["Form"] for mark in ("?", "(sg)", "(pl)")) for row in rows)
    assert all(row["Native"] == row["Parameter_ID"] == "" for row in rows)
    assert all(
        row[field] == ""
        for row in rows
        for field in (
            "Cognateset", "Etymology", "Variant_Of_Key", "Borrowed_From_Key",
            "Derivation_Parent_Keys",
        )
    )
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix A.3, printed p. ") for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)


def test_complete_audit_reconciles_source_responses_and_exclusions():
    audit = audited()
    assert len(audit) == 1765
    assert Counter(row["Status"] for row in audit) == Counter(installed=1538, excluded=227)
    assert Counter(row["Record_Type"] for row in audit) == Counter(
        {"wordlist response": 1760, "item exclusion": 5}
    )
    wordlists = [row for row in audit if row["Record_Type"] == "wordlist response"]
    assert {int(row["Gloss_Number"]) for row in wordlists} == set(range(1, 211)) - {11, 23, 32, 70, 188}
    assert len([row for row in wordlists if row["Comparison_Role"] == "comparison control"]) == 214
    assert all(
        row["Status"] == "excluded"
        for row in wordlists if row["Comparison_Role"] == "comparison control"
    )
    gaps = [row for row in wordlists if row["Reason"] == "source explicitly prints No Entry"]
    assert len(gaps) == 8 and all(row["Status"] == "excluded" for row in gaps)
    assert {(int(row["Gloss_Number"]), row["Site_Code"]) for row in gaps} >= {
        (208, "Reyavani"), (209, "Bobbili"),
    }
    disqualified = [row for row in audit if row["Record_Type"] == "item exclusion"]
    assert {int(row["Gloss_Number"]) for row in disqualified} == {11, 23, 32, 70, 188}
    assert not [row for row in audit if row["Status"] in {"unparsed", "unmapped"}]


def test_manifest_and_uncertainty_counts_are_pinned():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["publisher_file_sha256"] == (
        "f5fd88b84e1add2509314186bbde779e35e6675c96390c885d712ffee39b9300"
    )
    assert manifest["source_response_records"] == 1760
    assert manifest["target_installed"] == 1538
    assert manifest["target_no_entry"] == 8
    assert manifest["control_records"] == 214
    assert manifest["uncertainty_counts"] == {
        "source-raster-vowel": 720,
        "source-raster-coronal": 771,
        "source-raster-nasalization": 10,
        "source-raster-unresolved": 2,
        "source-raster-superscript": 67,
    }


def test_representative_entries_annotations_and_source_edges():
    by_key = {row["Entry_Key"]: row for row in forms()}
    stone = by_key["silgadaba2019:g052:sil-gadaba-2019-bobbilivalasa:i1"]
    assert stone["Form"] == stone["Phonemic"] == "kʌɳɖu"
    assert stone["Source"].endswith("printed p. 19, item 52, Bobbilivalasa]")

    plural = by_key["silgadaba2019:g018:sil-gadaba-2019-bobbilivalasa:i1"]
    assert plural["Form"] == "kɑlgil"
    assert plural["Phonemic"] == "kɑlgil(pl)"
    assert "source marks pl" in plural["Notes"]

    unresolved = by_key["silgadaba2019:g012:sil-gadaba-2019-panukuvalasa:i1"]
    assert unresolved["Form"] == "puɖʊ"
    assert unresolved["Phonemic"] == "puɖʊ?"
    assert "source-raster-unresolved" in unresolved["Notes"]

    assert not any(":g208:sil-gadaba-2019-reyavanivalasa:" in key for key in by_key)
    assert not any("Srikakulam" in row["Source"] for row in forms())


def test_source_and_all_seven_dialects_are_registered():
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    ids = {row["Tags"].split(":")[2] for row in forms()}
    assert ids == DIALECT_IDS
    for dialect_id in ids:
        assert dialects[dialect_id]["Language_ID"] == "Gadaba"
        assert dialects[dialect_id]["Glottocode"] == "mudh1235"
        assert dialects[dialect_id]["Quality"] == "C"
        assert "approximate mandal-centre coordinate" in dialects[dialect_id]["Location"]


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {part.split("[", 1)[0].strip() for part in row["Source"].split(";")}
        ]
    if not compiled:
        pytest.skip("cldf/forms.csv predates the Gadaba ingest; enforced after make all")
    assert len(compiled) == len(forms())
    assert {row["Language_ID"] for row in compiled} == {"Gadaba"}
    if "Status" in compiled[0]:
        assert all(row["Status"] == "unlinked" for row in compiled)
    else:
        assert all(row["Parameter_ID"] == row["Cognateset"] == "" for row in compiled)
    assert all(row["Original"] for row in compiled)
