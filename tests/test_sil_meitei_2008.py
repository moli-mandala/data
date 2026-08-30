"""Regression tests for SIL ESR 2008-002's SAG-IPA Meitei wordlists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_meitei_2008"
EXTRACTOR = SOURCE_DIR / "extract_meitei.py"
IMPORTER = SOURCE_DIR / "import_meitei.py"
TRANSCRIPTION = SOURCE_DIR / "wordlists.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-meitei.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-meitei-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-meitei-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
PDF = Path("/tmp/silesr2008_002.pdf")
SOURCE_KEY = "kim-kim2008meitei"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
DIALECT_IDS = {
    "sil-meitei-2008-mukabil", "sil-meitei-2008-humerjan",
    "sil-meitei-2008-shivganj", "sil-meitei-2008-shivnagar",
    "sil-meitei-2008-choto-dhamai", "sil-meitei-2008-kunagaon",
    "sil-meitei-2008-lilong-bazaar", "sil-meitei-2008-imphal",
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
    assert "installed=2406 controls=307 audit=2713" in result.stdout


@pytest.mark.skipif(not PDF.exists(), reason="official SIL PDF is not cached")
def test_official_pdf_extraction_reproduces_the_frozen_transcription(tmp_path):
    output = tmp_path / "wordlists.tsv"
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR), str(PDF), "--output", str(output)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    assert "items=307 printed_responses=1219 expanded=2713 legacy_glyphs=2534" in result.stdout
    assert output.read_bytes() == TRANSCRIPTION.read_bytes()


def test_transcription_topology_and_legacy_glyph_census_are_pinned():
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 1219
    assert {int(row["Item"]) for row in rows} == set(range(1, 308))
    assert sum(len(row["Site_Codes"]) for row in rows) == 2713
    assert Counter(code for row in rows for code in row["Site_Codes"]) == Counter(
        {"0": 307, "1": 291, "2": 298, "3": 295, "4": 296,
         "5": 298, "6": 300, "7": 317, "8": 311}
    )
    with (SOURCE_DIR / "sag_ipa_used.tsv").open(encoding="utf-8", newline="") as stream:
        glyphs = list(csv.DictReader(stream, delimiter="\t"))
    assert len(glyphs) == 25
    assert sum(int(row["Occurrences"]) for row in glyphs) == 2534
    assert {row["Glyph"].replace("◌", "") for row in glyphs} >= {
        "ː", "ɐ", "ʌ", "ʃ", "ʔ", "ŋ", "ɽ", "̯", "̃", "̥",
    }


def test_installed_scope_keys_and_source_transcription_contract():
    rows = forms()
    assert len(rows) == 2406
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert {row["Language_ID"] for row in rows} == {"Manipuri"}
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


def test_complete_audit_reconciles_target_and_control_records():
    audit = audited()
    assert len(audit) == 2713
    assert Counter(row["Status"] for row in audit) == Counter(installed=2406, excluded=307)
    assert {row["Record_Type"] for row in audit} == {"expanded wordlist attestation"}
    controls = [row for row in audit if row["Site_Code"] == "0"]
    assert len(controls) == 307
    assert all(row["Status"] == "excluded" and not row["Language_ID"] for row in controls)
    assert not [row for row in audit if row["Status"] in {"unparsed", "unmapped"}]


def test_manifest_representatives_source_language_and_dialects_are_registered():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["transcription_sha256"] == (
        "a8a1af7d6c8f418fe1e11e6a5628f726e3247a43fc2ecf98fb29f183df94db78"
    )
    assert manifest["unparsed_lines"] == manifest["unmapped_legacy_symbols"] == 0
    by_key = {row["Entry_Key"]: row for row in forms()}
    assert by_key["silmeitei2008:g001:sil-meitei-2008-mukabil:i1"]["Form"] == "asman"
    assert by_key["silmeitei2008:g164:sil-meitei-2008-shivnagar:i1"]["Form"] == "laŋʃoi̯"
    assert by_key["silmeitei2008:g307:sil-meitei-2008-kunagaon:i1"]["Form"] == "makhoi̯ ʃiŋ"

    assert f"@techreport{{{SOURCE_KEY}," in (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert languages["Manipuri"]["Glottocode"] == "mani1292"
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    ids = {row["Tags"].split(":")[2] for row in forms()}
    assert ids == DIALECT_IDS
    for dialect_id in ids:
        assert dialects[dialect_id]["Language_ID"] == "Manipuri"
        assert dialects[dialect_id]["Quality"] == "C"
    assert dialects["sil-meitei-2008-mukabil"]["Glottocode"] == "pang1284"


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {part.split("[", 1)[0].strip() for part in row["Source"].split(";")}
        ]
    if not compiled:
        pytest.skip("cldf/forms.csv predates the Meitei ingest; enforced after make all")
    assert len(compiled) == len(forms())
    assert {row["Language_ID"] for row in compiled} == {"Manipuri"}
    assert all(row["Original"] for row in compiled)
