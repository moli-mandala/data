"""Regression tests for SIL ESR 2018-011's Bareli/Pauri wordlists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_bareli_pauri_2018"
IMPORTER = SOURCE_DIR / "import_bareli_pauri.py"
SNAPSHOT = SOURCE_DIR / "wordlist_snapshot.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-bareli-pauri.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-bareli-pauri-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-bareli-pauri-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
SOURCE_KEY = "varkey-vunnamatla2018bareli"
LANGUAGE_GLOTTOCODES = {
    "RathwiBareli": "rath1242",
    "Bhilali": "bhil1253",
    "Bhili": "bhil1251",
    "Rathawi": "rath1243",
    "PalyaBareli": "paly1238",
    "PauriBareli": "paur1238",
    "Nimadi": "nima1243",
    "Khandesi": "khan1272",
}


def forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def audited():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_importer_rebuilds_checked_artifacts_without_pdf_or_pdfplumber():
    result = subprocess.run(
        [sys.executable, str(IMPORTER)], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert result.stdout.strip() == (
        "installed=6320 controls=789 no_entry=105 disqualified=33 audit=7247"
    )


def test_snapshot_topology_and_manifest_are_pinned():
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        snapshot = list(csv.DictReader(stream, delimiter="\t"))
    assert len(snapshot) == 7247
    assert {int(row["Concept"]) for row in snapshot} == set(range(1, 211))
    assert Counter(row["Source_Status"] for row in snapshot) == Counter(
        response=7214, disqualified=33
    )
    assert sum(row["Continuation"] == "1" for row in snapshot) == 317
    assert {row["Lect"] for row in snapshot if row["Concept"] != "70"} == {
        row["Lect"] for row in snapshot if row["Concept"] == "70"
    }

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_pdf_sha256"] == (
        "02128358a61e175ba2a07b2862f6072167a3609cf71264e235ae21284fe2ceea"
    )
    assert manifest["snapshot_sha256"] == (
        "92653e39eb64a96c30e4cc093b6fb6dd94fd8b3fd1890dfc2a3d76749c76d478"
    )
    assert manifest["counts"] == {
        "additional_response_lines": 317,
        "concepts": 210,
        "disqualified_concept_cells": 33,
        "excluded_records": 927,
        "explicit_no_entry_records": 105,
        "installed_regional_forms": 6320,
        "printed_response_records": 7214,
        "regional_lists": 30,
        "snapshot_and_audit_records": 7247,
        "standard_control_records": 789,
        "standard_controls": 3,
    }


def test_installed_scope_keys_and_transcription_contract():
    rows = forms()
    assert len(rows) == 6320
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert Counter(row["Language_ID"] for row in rows) == Counter(
        RathwiBareli=1469, Bhilali=1897, Bhili=842, Rathawi=206,
        PalyaBareli=434, PauriBareli=632, Nimadi=639, Khandesi=201,
    )
    assert len({row["Tags"].split(":")[2] for row in rows}) == 30
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
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix C.3, printed p. ") for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)


def test_complete_audit_reconciles_all_responses_and_exclusions():
    audit = audited()
    assert len(audit) == 7247
    assert Counter(row["Status"] for row in audit) == Counter(installed=6320, excluded=927)
    assert Counter(row["Reason"] for row in audit if row["Status"] == "excluded") == Counter({
        "standard-language comparison control": 789,
        "source explicitly says NO ENTRY": 105,
        "source marks item 70 DISQUALIFIED for every list": 33,
    })
    assert {int(row["Concept"]) for row in audit} == set(range(1, 211))
    millet = [row for row in audit if row["Concept"] == "70"]
    assert len(millet) == 33
    assert all(row["Status"] == "excluded" and row["Form"] == "DISQUALIFIED" for row in millet)
    controls = [row for row in audit if row["Reason"] == "standard-language comparison control"]
    assert {row["Lect"] for row in controls} == {"Hindi", "Gujarati", "Marathi"}
    assert not [row for row in audit if row["Status"] in {"unparsed", "unmapped"}]


def test_wrapped_alternatives_and_source_annotations_are_preserved():
    by_key = {row["Entry_Key"]: row for row in forms()}
    hungry = by_key[
        "silbareli2018:g184:sil-bareli-2018-rathwi-bareli-tharadpura:i1"
    ]
    assert hungry["Form"] == "bɦuklu tʃe, bɦuklu hʌtɔ̪"
    assert hungry["Source"].endswith("printed p. 140, item 184, Rathwi Bareli-Tharadpura]")

    assert by_key["silbareli2018:g001:sil-bareli-2018-bhili-kharod:i2"]["Form"] == "dil"
    five = by_key["silbareli2018:g155:sil-bareli-2018-rathwi-pauri-amalwadi:i1"]
    seven = by_key["silbareli2018:g157:sil-bareli-2018-bareli-pauri-mandvi:i1"]
    assert five["Form"] == "pats̪[" and seven["Form"] == "hat["
    assert "literal unmatched open bracket" in five["Notes"]
    assert "literal unmatched open bracket" in seven["Notes"]

    annotations = [row for row in forms() if "source annotation:" in row["Notes"]]
    assert Counter(row["Notes"].split("source annotation: ", 1)[1] for row in annotations) == Counter(
        {"big": 2, "small": 1, "on ground": 1, "mango tree": 1}
    )
    assert all(not any(token in row["Form"] for token in ("(big)", "(small)", "(on ground)", "(mango tree)")) for row in forms())

    marathi = next(
        row for row in audited()
        if row["Concept"] == "188" and row["Lect"] == "Marathi"
    )
    assert marathi["Form"] == "lek" and marathi["Status"] == "excluded"


def test_source_languages_and_all_thirty_dialects_are_registered():
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib

    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    for language, glottocode in LANGUAGE_GLOTTOCODES.items():
        assert languages[language]["Glottocode"] == glottocode
        assert languages[language]["Quality"] == "C"

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    ids = {row["Tags"].split(":")[2] for row in forms()}
    assert len(ids) == 30
    for dialect_id in ids:
        dialect = dialects[dialect_id]
        assert dialect["Glottocode"] == LANGUAGE_GLOTTOCODES[dialect["Language_ID"]]
        assert dialect["Quality"] == "C"
        assert dialect["Location"]
        assert dialect["Latitude"] == dialect["Longitude"] == ""


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build_with_original_ipa():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {part.split("[", 1)[0].strip() for part in row["Source"].split(";")}
        ]
    if not compiled:
        pytest.skip("cldf/forms.csv predates the Bareli/Pauri ingest; enforced after make all")
    assert len(compiled) == len(forms())
    assert {row["Language_ID"] for row in compiled} == set(LANGUAGE_GLOTTOCODES)
    assert all(row["Original"] for row in compiled)
    assert not any("�" in row["Form"] for row in compiled)
    hungry = next(
        row for row in compiled
        if row["Gloss"] == "he is, he was hungry"
        and row["Language_ID"] == "RathwiBareli"
        and row["Source"].endswith("item 184, Rathwi Bareli-Tharadpura]")
    )
    assert hungry["Form"] == "bɦuklu ce, bɦuklu hato"
    assert hungry["Original"] == "bɦuklu tʃe, bɦuklu hʌtɔ̪"
