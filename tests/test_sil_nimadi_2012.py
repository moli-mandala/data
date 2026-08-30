"""Regression tests for SIL ESR 2012-002's Nimadi wordlists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_nimadi_2012"
IMPORTER = SOURCE_DIR / "import_nimadi.py"
SNAPSHOT = SOURCE_DIR / "wordlist_snapshot.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-nimadi.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-nimadi-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-nimadi-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
SOURCE_KEY = "vunnamatla-john-samuvel2012nimadi"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]


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
        "installed=2826 comparisons=1207 no_entry=5 missing=2 omitted=52 audit=4092"
    )


def test_snapshot_topology_and_manifest_are_pinned():
    with SNAPSHOT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 4092
    assert {int(row["Concept"]) for row in rows} == set(range(1, 211))
    assert {row["Lect"] for row in rows} == {
        "N-Son-Bal", "N-Son-Pat", "N-Bal-Br", "N-Jaj-OBC", "N-Bhi-Bhi",
        "N-Dhar-Bhi", "N-Khj-Bhi", "N-Mah-Bhi", "N-Rup-Br", "N-Khr-Gen",
        "N-Awl-Bal", "N-Sir-OBC", "N-Kup-Dar", "Par Bhi", "Malvi", "Hindi",
        "Gujarati", "Marathi",
    }
    assert Counter(row["Source_Status"] for row in rows) == Counter(
        response=4018, blank=1, omitted_prompt=72, implicit_missing=1,
    )
    assert Counter(row["Response_Index"] for row in rows) == Counter(
        {"1": 3780, "2": 280, "3": 30, "4": 2}
    )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_pdf_sha256"] == (
        "1a7e8daaeb2b967e2f9490292689e33a188caf47dc262c942a47136bb270d0d8"
    )
    assert manifest["snapshot_sha256"] == (
        "a2822d380b0f195427056c0b5c6756a9cf5ae756eb49072ea411fd4802d44e08"
    )
    assert manifest["counts"] == {
        "additional_response_lines": 312,
        "comparison_lists": 5,
        "comparison_records": 1207,
        "excluded_records": 1266,
        "explicit_no_entry_records": 7,
        "implicit_missing_cells": 1,
        "installed_nimadi_forms": 2826,
        "printed_prompts": 206,
        "printed_response_records": 4019,
        "snapshot_and_audit_records": 4092,
        "standard_prompts": 210,
        "synthetic_omitted_prompt_cells": 72,
        "target_lists": 13,
    }


def test_installed_scope_keys_and_transcription_contract():
    rows = forms()
    assert len(rows) == 2826
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert {row["Language_ID"] for row in rows} == {"Nimadi"}
    assert len({row["Tags"].split(":")[2] for row in rows}) == 13
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
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix A, printed p. ") for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)


def test_complete_audit_reconciles_missing_prompts_controls_and_no_entries():
    rows = audited()
    assert len(rows) == 4092
    assert Counter(row["Status"] for row in rows) == Counter(installed=2826, excluded=1266)
    excluded = Counter(row["Reason"] for row in rows if row["Status"] == "excluded")
    assert excluded == Counter({
        "borrowed or standard comparison list": 1207,
        "prompt absent from the published appendix": 52,
        "source explicitly says no entry": 5,
        "no primary form printed for this lect/concept cell": 2,
    })
    omitted = [row for row in rows if row["Reason"] == "prompt absent from the published appendix"]
    assert {int(row["Concept"]) for row in omitted} == {11, 23, 24, 70}
    assert len(omitted) == 13 * 4
    controls = [row for row in rows if row["Reason"] == "borrowed or standard comparison list"]
    assert {row["Lect"] for row in controls} == {"Par Bhi", "Malvi", "Hindi", "Gujarati", "Marathi"}
    assert not [row for row in rows if row["Status"] in {"unparsed", "unmapped"}]


def test_image_checked_text_layer_artifacts_and_predicate_commas_are_preserved():
    audit = audited()
    fused = next(row for row in audit if row["Concept"] == "13" and row["Lect"] == "N-Son-Bal")
    assert fused["Form"] == "ct̪"
    assert "fused to the source glyph run" in fused["Notes"]

    ring = [row for row in audit if row["Concept"] == "40" and row["Lect"] == "N-Rup-Br"]
    assert [(row["Category"], row["Form"], row["Status"]) for row in ring] == [
        ("1", "", "excluded"), ("2", "mund̪i", "installed"),
    ]
    mosquito = next(row for row in audit if row["Concept"] == "98" and row["Lect"] == "Gujarati")
    assert mosquito["Form"] == "mətʃʰəɾə" and "cid:1" not in mosquito["Form"]
    assert "spurious PDF text-layer" in mosquito["Notes"]

    hungry = next(
        row for row in forms()
        if row["Gloss"] == "he is hungry, he was hungry"
        and "sonipura-balai" in row["Entry_Key"]
    )
    assert hungry["Form"] == "bʱuklʌgi, bʱuklʌgti̪tʰ̪i"


def test_source_and_all_thirteen_dialects_are_registered():
    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert languages["Nimadi"]["Glottocode"] == "nima1243"
    assert languages["Nimadi"]["Quality"] == "C"

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    ids = {row["Tags"].split(":")[2] for row in forms()}
    assert len(ids) == 13
    for dialect_id in ids:
        dialect = dialects[dialect_id]
        assert dialect["Language_ID"] == "Nimadi"
        assert dialect["Glottocode"] == "nima1243"
        assert dialect["Quality"] == "C"
        assert dialect["Location"]
        assert dialect["Latitude"] == dialect["Longitude"] == ""


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_build_with_original_ipa():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {part.split("[", 1)[0].strip() for part in row["Source"].split(";")}
        ]
    if not compiled:
        pytest.skip("cldf/forms.csv predates the Nimadi ingest; enforced after make all")
    assert len(compiled) == 2826
    assert all(row["Original"] and "�" not in row["Form"] for row in compiled)
    hungry = next(
        row for row in compiled
        if row["Gloss"] == "he is hungry, he was hungry"
        and row["Source"].endswith("item 184, N-Son-Bal]")
    )
    assert hungry["Form"] == "bhuklagi, bhuklagtithi"
    assert hungry["Original"] == "bʱuklʌgi, bʱuklʌgti̪tʰ̪i"
