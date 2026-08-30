"""Regression tests for SIL ESR 2008-003's legacy-font Bishnupriya wordlists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_bishnupriya_2008"
EXTRACTOR = SOURCE_DIR / "extract_bishnupriya.py"
IMPORTER = SOURCE_DIR / "import_bishnupriya.py"
TRANSCRIPTION = SOURCE_DIR / "wordlists.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-bishnupriya.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-bishnupriya-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-bishnupriya-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
SCAFFOLD = Path("/tmp/bishnupriya-slideshare.html")
SOURCE_KEY = "kim-kim2008bishnupriya"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
DIALECT_IDS = {
    "sil-bishnupriya-2008-tilakpur", "sil-bishnupriya-2008-soi-sri",
    "sil-bishnupriya-2008-gulerhaor", "sil-bishnupriya-2008-dhonitila",
    "sil-bishnupriya-2008-machimpur", "sil-bishnupriya-2008-madhapur",
}


def forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def audited():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_importer_rebuilds_checked_artifacts_without_web_source():
    result = subprocess.run(
        [sys.executable, str(IMPORTER)], cwd=ROOT, check=True,
        text=True, capture_output=True,
    )
    assert "installed=1801 controls=298 empty_prompts=9 audit=2108" in result.stdout


@pytest.mark.skipif(not SCAFFOLD.exists(), reason="public transcript scaffold is not cached")
def test_fixed_layout_extraction_reproduces_the_frozen_transcription(tmp_path):
    output = tmp_path / "wordlists.tsv"
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR), str(SCAFFOLD), "--output", str(output)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    assert "printed_responses=746 expanded=2099 empty_prompts=9" in result.stdout
    assert output.read_bytes() == TRANSCRIPTION.read_bytes()


def test_transcription_topology_and_legacy_glyph_census_are_pinned():
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 746
    assert sum(len(row["Site_Codes"]) for row in rows) == 2099
    assert sum(int(row["Aspiration_Markers"]) for row in rows) == 161
    assert {int(row["Item"]) for row in rows} == set(range(1, 308)) - {
        194, 218, 221, 222, 258, 259, 301, 303, 306,
    }
    assert Counter(code for row in rows for code in row["Site_Codes"]) == Counter(
        {"0": 298, "a": 291, "b": 297, "c": 306, "d": 304, "e": 309, "f": 294}
    )
    with (SOURCE_DIR / "slideshare_pua_used.tsv").open(
        encoding="utf-8", newline=""
    ) as stream:
        glyphs = list(csv.DictReader(stream, delimiter="\t"))
    assert len(glyphs) == 14
    assert sum(int(row["Occurrences"]) for row in glyphs) == 947
    assert {row["Glyph"].replace("◌", "") for row in glyphs} >= {
        "ʃ", "ɛ", "ɾ", "ɡ", "ɔ", "ŋ", "ʒ", "̯", "ɖ", "̃",
    }


def test_installed_scope_keys_and_source_transcription_contract():
    rows = forms()
    assert len(rows) == 1801
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert {row["Language_ID"] for row in rows} == {"Bishnupriya"}
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
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix B.3, printed p. ") for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)


def test_complete_audit_reconciles_controls_and_empty_prompts():
    audit = audited()
    assert len(audit) == 2108
    assert Counter(row["Status"] for row in audit) == Counter(installed=1801, excluded=307)
    assert Counter(row["Record_Type"] for row in audit) == Counter(
        {"expanded wordlist attestation": 2099, "empty prompt": 9}
    )
    controls = [row for row in audit if row["Site_Code"] == "0"]
    assert len(controls) == 298 and all(row["Status"] == "excluded" for row in controls)
    empty = [row for row in audit if row["Record_Type"] == "empty prompt"]
    assert {int(row["Gloss_Number"]) for row in empty} == {
        194, 218, 221, 222, 258, 259, 301, 303, 306,
    }
    typo = [row for row in controls if "lowercase o" in row["Uncertainty"]]
    assert len(typo) == 1 and typo[0]["Gloss_Number"] == "267"
    assert not [row for row in audit if row["Status"] in {"unparsed", "unmapped"}]


def test_manifest_representatives_source_language_and_dialects_are_registered():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["wordlist_transcript_sha256"] == (
        "1c42bad6a4ee278b4056397f3c5db960b091d5e1df749ff021137a0398aeac8b"
    )
    assert manifest["unparsed_lines"] == manifest["unmapped_legacy_symbols"] == 0
    by_key = {row["Entry_Key"]: row for row in forms()}
    assert by_key["silbishnupriya2008:g001:sil-bishnupriya-2008-tilakpur:i1"]["Form"] == "dɪn"
    assert by_key["silbishnupriya2008:g138:sil-bishnupriya-2008-soi-sri:i1"]["Form"] == "hurkaŋbɛjok"
    assert by_key["silbishnupriya2008:g267:sil-bishnupriya-2008-machimpur:i2"]["Form"] == "lai̯ lai̯"

    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert languages["Bishnupriya"]["Glottocode"] == "bish1244"
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    ids = {row["Tags"].split(":")[2] for row in forms()}
    assert ids == DIALECT_IDS
    for dialect_id in ids:
        assert dialects[dialect_id]["Language_ID"] == "Bishnupriya"
        assert dialects[dialect_id]["Glottocode"] == "bish1244"
        assert dialects[dialect_id]["Quality"] == "C"


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {part.split("[", 1)[0].strip() for part in row["Source"].split(";")}
        ]
    if not compiled:
        pytest.skip("cldf/forms.csv predates the Bishnupriya ingest; enforced after make all")
    assert len(compiled) == len(forms())
    assert {row["Language_ID"] for row in compiled} == {"Bishnupriya"}
    assert all(row["Original"] for row in compiled)
