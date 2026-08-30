"""Regression tests for SIL ESR 2008-013's legacy-SAG Jaunsari wordlists."""

import csv
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/sil_jaunsari_2008"
IMPORTER = SOURCE_DIR / "import_jaunsari.py"
EXTRACTOR = SOURCE_DIR / "extract_jaunsari.py"
TRANSCRIPTION = SOURCE_DIR / "wordlists.tsv"
INSTALLED = ROOT / "data/other/forms/20260828-sil-jaunsari.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-jaunsari-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-jaunsari-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
SOURCE_PDF = ROOT.parent / "tmp/pdfs/sil-surveys/silesr2008_013.pdf"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
SOURCE_KEY = "john2008jaunsari"
DIALECT_IDS = {
    "sil-jaunsari-2008-chakrata",
    "sil-jaunsari-2008-bhandroli",
    "sil-jaunsari-2008-chapnu",
    "sil-jaunsari-2008-khanaad",
    "sil-jaunsari-2008-korwa",
    "sil-jaunsari-2008-lakhamandal",
    "sil-jaunsari-2008-maindrath",
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
    assert "installed=1619" in result.stdout
    assert "controls=1110" in result.stdout
    assert "audit=2732" in result.stdout


@pytest.mark.skipif(not SOURCE_PDF.exists(), reason="publisher PDF is outside the data repository")
def test_official_legacy_decode_reproduces_the_frozen_transcription(tmp_path):
    output = tmp_path / "wordlists.tsv"
    result = subprocess.run(
        [sys.executable, str(EXTRACTOR), str(SOURCE_PDF), "--output", str(output)],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    assert "responses=2729 target=1619 controls=1110" in result.stdout
    assert output.read_bytes() == TRANSCRIPTION.read_bytes()


def test_transcription_topology_and_authoritative_map_subset_are_pinned():
    with TRANSCRIPTION.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 2729
    assert Counter(row["Role"] for row in rows) == Counter(
        {"target": 1619, "comparison control": 1110}
    )
    assert {int(row["Item"]) for row in rows} == set(range(1, 211)) - {11, 23, 24}
    assert {row["Source_Code"] for row in rows} == set("ABCDKLMhS JNG".replace(" ", ""))
    with (SOURCE_DIR / "sag_ipa_used.tsv").open(encoding="utf-8", newline="") as stream:
        mapping = list(csv.DictReader(stream, delimiter="\t"))
    assert len(mapping) == 32
    assert {row["Byte"] for row in mapping} >= {"43", "84", "98", "99", "E6"}


def test_installed_scope_keys_and_transcription_contract():
    rows = forms()
    assert len(rows) == 1619
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert {row["Language_ID"] for row in rows} == {"jaun"}
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
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Appendix A.2, printed p. ") for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)


def test_complete_audit_reconciles_controls_and_source_exclusions():
    audit = audited()
    assert len(audit) == 2732
    assert Counter(row["Status"] for row in audit) == Counter(installed=1619, excluded=1113)
    assert Counter(row["Record_Type"] for row in audit) == Counter(
        {"wordlist response": 2729, "item exclusion": 3}
    )
    controls = [row for row in audit if row["Comparison_Role"] == "comparison control"]
    assert len(controls) == 1110 and all(row["Status"] == "excluded" for row in controls)
    exclusions = [row for row in audit if row["Record_Type"] == "item exclusion"]
    assert {int(row["Gloss_Number"]) for row in exclusions} == {11, 23, 24}
    assert not [row for row in audit if row["Status"] in {"unparsed", "unmapped"}]


def test_manifest_representatives_source_and_dialects_are_registered():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["publisher_file_sha256"] == (
        "e6b3b6d54c061d03614b27618f0f06d2138f07c47dc1a266d45b0fe16bd75f68"
    )
    assert manifest["official_converter_map_sha256"] == (
        "a989926e91d4b562df20758cbb613f0177fce33d1c2e9e02195087e94f1f2930"
    )
    assert manifest["unparsed_lines"] == manifest["unmapped_legacy_symbols"] == 0

    by_key = {row["Entry_Key"]: row for row in forms()}
    assert by_key["siljaunsari2008:g001:sil-jaunsari-2008-korwa:i1"]["Form"] == "çʌɾiɾ"
    assert by_key["siljaunsari2008:g009:sil-jaunsari-2008-korwa:i1"]["Form"] == "d̪ant̪"
    assert by_key["siljaunsari2008:g210:sil-jaunsari-2008-maindrath:i2"]["Form"] == "jɛ dʒɛ"

    bib = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bib
    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    ids = {row["Tags"].split(":")[2] for row in forms()}
    assert ids == DIALECT_IDS
    for dialect_id in ids:
        assert dialects[dialect_id]["Language_ID"] == "jaun"
        assert dialects[dialect_id]["Glottocode"] == "jaun1243"
        assert dialects[dialect_id]["Quality"] == "C"


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_installed_row_survives_the_full_build():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {part.split("[", 1)[0].strip() for part in row["Source"].split(";")}
        ]
    if not compiled:
        pytest.skip("cldf/forms.csv predates the Jaunsari ingest; enforced after make all")
    assert len(compiled) == len(forms())
    assert {row["Language_ID"] for row in compiled} == {"jaun"}
    assert all(row["Original"] for row in compiled)
