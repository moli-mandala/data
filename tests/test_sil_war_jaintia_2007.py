"""Regression tests for SIL ESR 2007-013's SAG-IPA War-Jaintia wordlists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_war_jaintia_2007"
EXTRACTOR = SOURCE_DIR / "extract_war_jaintia.py"
IMPORTER = SOURCE_DIR / "import_war_jaintia.py"
TRANSCRIPTION = SOURCE_DIR / "wordlists.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-war-jaintia.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-war-jaintia-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-war-jaintia-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
PDF = Path("/tmp/silesr2007_013-war-jaintia.pdf")
SOURCE_KEY = "brightbill-kim-kim2007warjaintia"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
DIALECT_IDS = {
    "sil-war-jaintia-2007-niralapunji", "sil-war-jaintia-2007-aliachora",
    "sil-war-jaintia-2007-dabolchora", "sil-war-jaintia-2007-singur",
    "sil-war-jaintia-2007-barenga", "sil-war-jaintia-2007-magurchora",
    "sil-war-jaintia-2007-amlarem",
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
    assert "installed=2030 controls=1428 undefined=1 audit=3459" in result.stdout


@pytest.mark.skipif(not PDF.exists(), reason="preserved official SIL PDF is not cached")
def test_official_pdf_extraction_reproduces_the_frozen_transcription(tmp_path):
    output = tmp_path / "wordlists.tsv"
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR), str(PDF), "--output", str(output)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    assert "items=307 printed_responses=1690 expanded=3459 legacy_glyphs=2398" in result.stdout
    assert output.read_bytes() == TRANSCRIPTION.read_bytes()


def test_transcription_topology_and_legacy_glyph_census_are_pinned():
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 1690
    assert {int(row["Item"]) for row in rows} == set(range(1, 308)) - {
        21, 30, 36, 39, 40, 41, 42, 51, 60, 64, 65, 67, 72, 75, 88,
        124, 146, 149, 159, 168, 171, 194, 199, 203, 209, 239, 240, 255,
        257, 301, 306,
    }
    assert sum(len(row["Site_Codes"]) for row in rows) == 3459
    assert Counter(code for row in rows for code in row["Site_Codes"]) == Counter({
        "A": 288, "B": 292, "C": 283, "D": 287, "E": 293, "F": 288,
        "G": 282, "H": 277, "I": 293, "J": 294, "K": 289, "L": 292,
        "U": 1,
    })
    with (SOURCE_DIR / "sag_ipa_used.tsv").open(encoding="utf-8", newline="") as stream:
        glyphs = list(csv.DictReader(stream, delimiter="\t"))
    assert len(glyphs) == 17
    assert sum(int(row["Occurrences"]) for row in glyphs) == 2398
    assert {row["Glyph"].replace("◌", "") for row in glyphs} >= {
        "ɨ", "ŋ", "ʃ", "ʔ", "ɔ", "ʈ", "ɲ", "̯", "̃",
    }


def test_installed_scope_keys_and_source_transcription_contract():
    rows = forms()
    assert len(rows) == 2030
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert {row["Language_ID"] for row in rows} == {"WarJaintia"}
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


def test_complete_audit_reconciles_targets_controls_and_source_anomaly():
    audit = audited()
    assert len(audit) == 3459
    assert Counter(row["Status"] for row in audit) == Counter(installed=2030, excluded=1429)
    assert {row["Record_Type"] for row in audit} == {"expanded wordlist attestation"}
    anomaly = [row for row in audit if row["Site_Code"] == "U"]
    assert len(anomaly) == 1
    assert anomaly[0]["Gloss_Number"] == "119" and anomaly[0]["Status"] == "excluded"
    group_a = [
        row for row in audit
        if row["Gloss_Number"] == "137" and row["Site_Code"] == "K"
    ]
    assert len(group_a) == 1 and group_a[0]["Similarity_Group"] == "A"
    assert not [row for row in audit if row["Status"] in {"unparsed", "unmapped"}]


def test_manifest_representatives_source_language_and_dialects_are_registered():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["transcription_sha256"] == (
        "d4bb84edf77b95a66b71a0cb7e1915365d21d778a0a0dd39d4ba0a855a84db37"
    )
    assert manifest["unparsed_lines"] == manifest["unmapped_legacy_symbols"] == 0
    by_key = {row["Entry_Key"]: row for row in forms()}
    assert by_key["silwarjaintia2007:g001:sil-war-jaintia-2007-niralapunji:i1"]["Form"] == "phli jaŋ"
    assert by_key["silwarjaintia2007:g119:sil-war-jaintia-2007-aliachora:i1"]["Form"] == "lɔ ɔt"
    assert by_key["silwarjaintia2007:g307:sil-war-jaintia-2007-amlarem:i1"]["Form"] == "i ja"

    assert f"@techreport{{{SOURCE_KEY}," in (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert languages["WarJaintia"]["Glottocode"] == "warj1242"
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    ids = {row["Tags"].split(":")[2] for row in forms()}
    assert ids == DIALECT_IDS
    for dialect_id in ids:
        assert dialects[dialect_id]["Language_ID"] == "WarJaintia"
        assert dialects[dialect_id]["Glottocode"] == "warj1242"
        assert dialects[dialect_id]["Quality"] == "C"


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {part.split("[", 1)[0].strip() for part in row["Source"].split(";")}
        ]
    if not compiled:
        pytest.skip("cldf/forms.csv predates the War-Jaintia ingest; enforced after make all")
    assert len(compiled) == len(forms())
    assert {row["Language_ID"] for row in compiled} == {"WarJaintia"}
    assert all(row["Original"] for row in compiled)
