"""Regression tests for SIL ESR 2011-025's Bangladesh Kuki-Chin wordlists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_kuki_chin_bangladesh_2011"
EXTRACTOR = SOURCE_DIR / "extract_kuki_chin.py"
IMPORTER = SOURCE_DIR / "import_kuki_chin.py"
TRANSCRIPTION = SOURCE_DIR / "wordlists.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-kuki-chin-bangladesh.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-kuki-chin-bangladesh-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-kuki-chin-bangladesh-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
PDF = Path("/tmp/kuki-chin-appendix-a.pdf")
SOURCE_KEY = "kim-roy-sangma2011kukichin"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
LANGUAGES = {
    "Pangkhua": "pank1249", "BawmChin": "bawm1236", "AshoChin": "asho1236",
    "KhumiChin": "khum1248", "Mizo": "lush1249",
}
DIALECT_IDS = {
    "sil-kuki-chin-2011-bilaichari-pangkhua", "sil-kuki-chin-2011-konglak",
    "sil-kuki-chin-2011-bethel-para-bawm", "sil-kuki-chin-2011-jamunachari",
    "sil-kuki-chin-2011-bethel-para-mizo", "sil-kuki-chin-2011-mahmuam-para",
    "sil-kuki-chin-2011-boro-kukyachari", "sil-kuki-chin-2011-ghungurumukh-para",
    "sil-kuki-chin-2011-manglung-headman-para", "sil-kuki-chin-2011-prongphung-para",
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
    assert "installed=3235 bangla_controls=307 myanmar_khumi_controls=333 audit=3875" in result.stdout


@pytest.mark.skipif(not PDF.exists(), reason="public SIL appendix PDF is not cached")
def test_public_pdf_extraction_reproduces_the_frozen_transcription(tmp_path):
    output = tmp_path / "wordlists.tsv"
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR), str(PDF), "--output", str(output)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    assert "items=306 printed_responses=2565 expanded=3875 no_entry=53" in result.stdout
    assert output.read_bytes() == TRANSCRIPTION.read_bytes()


def test_transcription_topology_and_legacy_glyph_census_are_pinned():
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 2565
    assert {int(row["Item"]) for row in rows} == set(range(1, 307))
    assert sum(len(row["Site_Codes"]) for row in rows) == 3875
    assert Counter(code for row in rows for code in row["Site_Codes"]) == Counter({
        "e": 333, "c": 329, "h": 327, "l": 324, "i": 324, "m": 324,
        "a": 323, "b": 323, "j": 321, "g": 321, "k": 319, "0": 307,
    })
    with (SOURCE_DIR / "sag_ipa_used.tsv").open(encoding="utf-8", newline="") as stream:
        glyphs = list(csv.DictReader(stream, delimiter="\t"))
    assert len(glyphs) == 65
    assert sum(int(row["Occurrences"]) for row in glyphs) == 16029
    assert {row["Glyph"].replace("◌", "") for row in glyphs} >= {
        "ɨ", "ŋ", "ʃ", "ʔ", "ɔ", "ʈ", "ɲ", "ɬ", "ɶ", "̯", "̚", "̩",
    }


def test_installed_scope_keys_languages_and_source_transcription_contract():
    rows = forms()
    assert len(rows) == 3235
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert Counter(row["Language_ID"] for row in rows) == Counter({
        "AshoChin": 648, "KhumiChin": 648, "Pangkhua": 647,
        "BawmChin": 647, "Mizo": 645,
    })
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
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix A.3, printed p. ") for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)


def test_complete_audit_reconciles_targets_and_controls():
    audit = audited()
    assert len(audit) == 3875
    assert Counter(row["Status"] for row in audit) == Counter(installed=3235, excluded=640)
    assert {row["Record_Type"] for row in audit} == {"expanded wordlist attestation"}
    bangla = [row for row in audit if row["Site_Code"] == "0"]
    myanmar = [row for row in audit if row["Site_Code"] == "e"]
    assert len(bangla) == 307 and len(myanmar) == 333
    assert sum(not row["Transcription"] for row in myanmar) == 53
    assert all(row["Status"] == "excluded" and not row["Language_ID"] for row in bangla + myanmar)
    assert not [row for row in audit if row["Status"] in {"unparsed", "unmapped"}]


def test_manifest_representatives_source_languages_and_dialects_are_registered():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["transcription_sha256"] == (
        "8d76f3d8f0345ad103cc08daf391753e62b6d7f5f704f9f73261fb5e3ce18438"
    )
    assert manifest["unparsed_lines"] == manifest["unmapped_legacy_symbols"] == 0
    by_key = {row["Entry_Key"]: row for row in forms()}
    assert by_key["silkukichin2011:g001:sil-kuki-chin-2011-bilaichari-pangkhua:i1"]["Form"] == "rɨvan"
    assert by_key["silkukichin2011:g119:sil-kuki-chin-2011-boro-kukyachari:i1"]["Form"] == "əthɔu̯"
    assert by_key["silkukichin2011:g306:sil-kuki-chin-2011-mahmuam-para:i1"]["Form"] == "anmaʔni"

    assert f"@techreport{{{SOURCE_KEY}," in (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    for language_id, glottocode in LANGUAGES.items():
        assert languages[language_id]["Glottocode"] == glottocode
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    ids = {row["Tags"].split(":")[2] for row in forms()}
    assert ids == DIALECT_IDS
    for dialect_id in ids:
        assert dialects[dialect_id]["Language_ID"] in LANGUAGES
        assert dialects[dialect_id]["Quality"] == "C"


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {part.split("[", 1)[0].strip() for part in row["Source"].split(";")}
        ]
    if not compiled:
        pytest.skip("cldf/forms.csv predates the Kuki-Chin ingest; enforced after make all")
    assert len(compiled) == len(forms())
    assert {row["Language_ID"] for row in compiled} == set(LANGUAGES)
    assert all(row["Original"] for row in compiled)
