"""Regression tests for SIL ESR 2018-009's Western Arunachal wordlists."""

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from segments import Tokenizer


ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data/other/forms/raw_data/abraham_monpa_2018"
IMPORTER = SOURCE_DIR / "import_abraham_monpa.py"
SNAPSHOT = SOURCE_DIR / "snapshot"
INSTALLED = ROOT / "data/other/forms/20260828-sil-western-arunachal-monpa.csv"
AUDIT = ROOT / "data/other/forms/raw_data/20260828-sil-western-arunachal-monpa-audit.csv"
MANIFEST = ROOT / "data/other/forms/raw_data/20260828-sil-western-arunachal-monpa-manifest.json"
COMPILED = ROOT / "cldf/forms.csv"
SOURCE_KEY = "abraham-sako-kinny-zeliang2018"
FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic",
    "Notes", "Source", "Cognateset", "Etymology", "Entry_Key",
    "Variant_Of_Key", "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
PARENT_COUNTS = Counter({
    "Bugun": 1587,
    "Miji": 1547,
    "KalaktangMonpa": 1250,
    "Tshangla": 1235,
    "Dakpakha": 925,
    "Sherdukpen": 617,
    "Sartang": 611,
    "Hruso": 308,
    "Khoina": 303,
    "Lish": 303,
    "Khoitam": 297,
    "Chug": 296,
})


def load_importer():
    spec = importlib.util.spec_from_file_location("abraham_monpa_importer", IMPORTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def forms():
    with INSTALLED.open(encoding="utf-8", newline="") as stream:
        return [dict(zip(FORM_FIELDS, row)) for row in csv.reader(stream)]


def audited():
    with AUDIT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def convert(form: str) -> str:
    profile = Tokenizer(str(ROOT / "conversion/tagin-puroik.txt"))
    # make_cldf's generic source-profile route tokenizes this source in NFC.
    converted = profile(unicodedata.normalize("NFC", form), column="IPA")
    return unicodedata.normalize("NFC", converted.replace(" ", "").replace("#", " "))


def test_importer_rebuilds_all_artifacts_offline():
    result = subprocess.run(
        [sys.executable, str(IMPORTER)], cwd=ROOT,
        check=True, text=True, capture_output=True,
    )
    assert (
        "installed=9279 audit=9421 gaps=142 "
        "recovered_beyond_upstream_cldf=1066"
    ) in result.stdout


def test_frozen_cc_by_snapshot_and_release_identity_are_pinned():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["upstream_release"] == "lexibank/abrahammonpa v3.0"
    assert manifest["upstream_release_doi"] == "10.5281/zenodo.5115885"
    assert manifest["upstream_release_zip_sha256"] == (
        "09a930bb46d1b43c512e83dbc13d7ebd30710a7ecdd895114b27f159bab60fbb"
    )
    assert manifest["license"] == "CC-BY-4.0"
    assert manifest["source_concepts"] == 307
    assert manifest["source_lects"] == 30
    assert manifest["source_cells"] == {
        "all": 9210,
        "filled": 9068,
        "gaps": 142,
        "multi_form": 210,
        "trailing_annotation": 2,
    }
    assert manifest["unparsed_rows"] == 0
    assert manifest["unmapped_concepts"] == 0
    assert manifest["unmapped_lects"] == 0
    for name, expected in manifest["snapshot_sha256"].items():
        assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == expected


def test_install_is_complete_source_faithful_and_graph_neutral():
    rows = forms()
    assert len(rows) == 9279
    assert {len(row) for row in csv.reader(INSTALLED.open(encoding="utf-8"))} == {15}
    assert Counter(row["Language_ID"] for row in rows) == PARENT_COUNTS
    assert len({row["Entry_Key"] for row in rows}) == len(rows)
    assert all(row["Form"] == row["Phonemic"] and row["Form"] for row in rows)
    assert all(unicodedata.normalize("NFC", row["Form"]) == row["Form"] for row in rows)
    assert all(row["Source"].startswith(f"{SOURCE_KEY}[Lexibank v3.0, ") for row in rows)
    assert all(row["Parameter_ID"] == row["Native"] == "" for row in rows)
    assert all(
        row[field] == ""
        for row in rows
        for field in (
            "Cognateset", "Etymology", "Variant_Of_Key", "Borrowed_From_Key",
            "Derivation_Parent_Keys",
        )
    )


def test_complete_audit_reconciles_every_source_cell_and_alternative():
    rows = audited()
    assert len(rows) == 9421
    assert Counter(row["Status"] for row in rows) == Counter(installed=9279, excluded=142)
    assert len({
        (row["Snapshot_Table"], row["Source_Row"], row["Source_Lect"])
        for row in rows
    }) == 9210
    assert {int(row["Concept_Number"]) for row in rows} == set(range(1, 308))
    gaps = [row for row in rows if row["Status"] == "excluded"]
    assert all(not row["Transcription"] and not row["Entry_Key"] for row in gaps)
    assert all(row["Reason"] == "explicit dash or blank source cell" for row in gaps)
    annotations = [row for row in rows if "trailing annotation" in row["Reason"]]
    assert len(annotations) == 2
    assert {row["Transcription"] for row in annotations} == {"t̪oŋkaŋmala", "zimbu"}
    assert all(" || ̃ai za" in row["Raw_Cell"] for row in annotations)
    assert not [row for row in rows if row["Status"] in {"unparsed", "unmapped"}]


def test_source_matrix_recovery_fixes_both_upstream_cldf_loss_modes():
    rows = {row["Entry_Key"]: row for row in forms()}
    # Lexibank's exact-name concept lookup overwrote source concept 81 with the
    # second identically labelled concept.  Both senses remain distinct here.
    assert rows["abrahammonpa2018:monpa:r082:f:v1"]["Gloss"] == "fat (organic substance)"
    assert rows["abrahammonpa2018:monpa:r082:f:v1"]["Form"] == "tʃhi"
    assert rows["abrahammonpa2018:monpa:r083:f:v1"]["Gloss"] == "fat (obese)"
    assert rows["abrahammonpa2018:monpa:r083:f:v1"]["Form"] == "phiak"
    # The source Kho-Bwa matrix writes this heading without a space; those rows
    # failed upstream's exact label lookup but are position- and label-resolved.
    assert rows["abrahammonpa2018:khobwa:r053:w:v1"]["Gloss"] == "cooked rice"
    assert rows["abrahammonpa2018:khobwa:r053:w:v1"]["Form"] == "mətʃije"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["upstream_cldf_forms"] == 8213
    assert manifest["forms_recovered_beyond_upstream_cldf"] == 1066


def test_all_parents_and_thirty_survey_lects_are_registered():
    module = load_importer()
    with (ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        languages = {row["ID"]: row for row in csv.DictReader(stream)}
    assert set(PARENT_COUNTS) <= set(languages)
    assert languages["KalaktangMonpa"]["Glottocode"] == "kala1376"
    assert languages["Tshangla"]["Glottocode"] == "tsha1245"
    assert languages["Dakpakha"]["Glottocode"] == "dakp1242"

    with (ROOT / "cldf/dialects.csv").open(encoding="utf-8", newline="") as stream:
        dialects = {row["ID"]: row for row in csv.DictReader(stream)}
    dialect_ids = {row["Tags"].split(":")[2] for row in forms()}
    assert len(dialect_ids) == 30
    assert dialect_ids == {module.dialect_id(upstream_id) for upstream_id in module.LECTS}
    assert all(dialects[dialect_id]["Quality"] == "C" for dialect_id in dialect_ids)
    assert all(dialects[dialect_id]["Latitude"] for dialect_id in dialect_ids)
    assert all(dialects[dialect_id]["Longitude"] for dialect_id in dialect_ids)

    bibliography = (ROOT / "cldf/sources.bib").read_text(encoding="utf-8")
    assert f"@techreport{{{SOURCE_KEY}," in bibliography
    assert "number       = {2018-009}" in bibliography


def test_western_arunachal_profile_covers_every_source_form():
    rows = forms()
    converted = [convert(row["Form"]) for row in rows]
    assert not any("�" in form for form in converted)
    assert convert("hasok1a") == "hasokla"
    assert convert("xol̪o") == "xolo"
    assert convert("ʔõː") == "ʔō̃"
    assert convert("ʃah̩") == "śah̩"


@pytest.mark.skipif(not COMPILED.exists(), reason="cldf/forms.csv has not been built")
def test_every_source_citation_survives_the_full_build():
    with COMPILED.open(encoding="utf-8", newline="") as stream:
        compiled = [
            row for row in csv.DictReader(stream)
            if SOURCE_KEY in {
                part.split("[", 1)[0].strip() for part in row["Source"].split(";")
            }
        ]
    if not compiled:
        pytest.skip("cldf/forms.csv predates the Western Arunachal ingest; enforced after make all")
    # Five Bugun lists substantially overlap Abraham & Sako (2021), so exact
    # duplicates may coalesce. Every installed source locator must still be
    # present on either its own or a merged compiled form.
    source_locators = {
        row["Source"] for row in forms()
    }
    compiled_locators = {
        citation.strip()
        for row in compiled
        for citation in row["Source"].split(";")
        if citation.strip().startswith(SOURCE_KEY)
    }
    assert compiled_locators == source_locators
    assert {row["Language_ID"] for row in compiled} == set(PARENT_COUNTS)
    assert all(row["Original"] for row in compiled)
