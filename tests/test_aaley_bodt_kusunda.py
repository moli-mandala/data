"""Regression tests for the Aaley--Bodt Kusunda 250-concept dataset."""

import csv
import hashlib
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from segments import Tokenizer


ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "data/other/forms/raw_data/aaley_bodt_kusunda_2020"
SNAPSHOT = PACKAGE / "snapshot"
IMPORTER = PACKAGE / "import_kusunda.py"
INSTALLED = ROOT / "data/other/forms/20260901-aaley-bodt-kusunda.csv"
AUDIT = PACKAGE / "20260901-aaley-bodt-kusunda-audit.csv"
SOURCE_KEY = "aaley-bodt2020kusunda"

SPEC = importlib.util.spec_from_file_location("aaley_bodt_kusunda", IMPORTER)
assert SPEC and SPEC.loader
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


def dicts(path: Path, *, delimiter=","):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_pinned_v21_release_and_checksums():
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["upstream_release"] == "lexibank/aaleykusunda v2.1"
    assert manifest["upstream_release_doi"] == "10.5281/zenodo.13149034"
    assert manifest["upstream_commit"] == "09b1e8d0c19e4f9c69352e4abee500212830f396"
    assert manifest["license"] == "CC-BY-4.0"
    source.verify_snapshot()
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((SNAPSHOT / filename).read_bytes()).hexdigest() == expected


def test_all_prompts_cells_and_released_lexemes_are_accounted_for():
    raw = dicts(SNAPSHOT / "Kusunda_2019_250_lexical_items.tsv", delimiter="\t")
    upstream = dicts(SNAPSHOT / "forms.csv")
    audit = dicts(AUDIT)
    installed = rows(INSTALLED)

    assert len(raw) == 250
    assert [row["ID"] for row in raw] == [str(i) for i in range(1, 251)]
    assert len(upstream) == len(installed) == 662
    assert len(audit) == 750
    assert Counter(row["Status"] for row in audit) == Counter(ingested=662, skipped=88)
    assert Counter(row["Source_Lect"] for row in audit) == Counter(
        ProtoKusunda=250, KusundaGM=250, KusundaK=250
    )
    assert Counter(row["Source_Lect"] for row in audit if row["Status"] == "ingested") == Counter(
        ProtoKusunda=211, KusundaGM=224, KusundaK=227
    )


def test_audit_preserves_raw_evidence_and_seeded_review():
    audit = {row["Audit_ID"]: row for row in dicts(AUDIT)}
    above = audit["1:KusundaGM"]
    assert above["Raw_Value"] == "nɔŋ.ʣeː ɐŋ.ʣeː"
    assert above["Upstream_Value"] == "nɔŋ.ʣeː ɐŋ.ʣeː"
    assert above["Upstream_Form"] == "ɐŋ.ʣeː"
    assert above["Upstream_Segments"] == "ɐ ŋ + dz eː"
    assert above["Entry_Key"] == "KusundaGM-1_above-1"
    assert audit["59:ProtoKusunda"]["Status"] == "skipped"
    assert audit["59:ProtoKusunda"]["Raw_Value"] == "∅"
    reviewed = [row for row in audit.values() if row["Review_State"] == "verified-no-material-error"]
    assert len(reviewed) == 20
    assert all(row["Status"] == "ingested" for row in reviewed)


def test_installed_rows_use_one_canonical_language_and_stable_keys():
    installed = rows(INSTALLED)
    assert all(len(row) == 15 for row in installed)
    assert {row[0] for row in installed} == {"Kusunda"}
    assert {row[1] for row in installed} == {""}
    assert len({row[10] for row in installed}) == len(installed)
    assert all(row[7].startswith(f"{SOURCE_KEY}[") for row in installed)
    assert all("dialect:" not in row[14] for row in installed)
    assert all(unicodedata.normalize("NFC", value) == value for row in installed for value in row)

    by_key = {row[10]: row for row in installed}
    assert by_key["KusundaGM-1_above-1"][2] == "ɐŋ.ʣeː"
    assert by_key["KusundaGM-1_above-1"][5] == "ɐŋ.ʣeː"
    assert "Full source value: nɔŋ.ʣeː ɐŋ.ʣeː" in by_key["KusundaGM-1_above-1"][6]
    assert "speaker Gyani Maiya Sen Kusunda" in by_key["KusundaGM-1_above-1"][7]
    assert set(by_key["KusundaGM-21_toburnintransitive-1"][14].split()) >= {"verb", "intr"}
    assert set(by_key["ProtoKusunda-48_eight-1"][14].split()) >= {
        "num", "loanword", "loan:Nepali"
    }
    assert set(by_key["KusundaGM-101_ifirstpersonsingular-1"][14].split()) >= {"pron", "1sg"}


def test_profile_covers_every_released_form_and_difficult_mappings():
    tokenizer = Tokenizer(str(ROOT / "conversion/kusunda-aaley-bodt.txt"))
    upstream = dicts(SNAPSHOT / "forms.csv")

    def convert(value):
        return unicodedata.normalize(
            "NFC", tokenizer(unicodedata.normalize("NFC", value), column="IPA")
            .replace(" ", "")
            .replace("#", " ")
        )

    converted = [(row["Form"], convert(row["Form"])) for row in upstream]
    assert not [(original, display) for original, display in converted if "�" in display]
    assert convert("ɐ̃ː.ʤi") == "ɐ̄̃ji"
    assert convert("mʲɛ̰kʰ") == "mʸɛ̰kʰ"
    assert convert("d̪əj.ʤiː") == "dəyjī"
    assert convert("ɐɴ.ʣe") == "ɐɴʣe"


def test_language_metadata_and_speaker_modeling():
    languages = {row["ID"]: row for row in dicts(ROOT / "cldf/languages.csv")}
    kusunda = languages["Kusunda"]
    assert kusunda["Glottocode"] == "kusu1250"
    assert (kusunda["Latitude"], kusunda["Longitude"]) == ("28.0", "82.26")
    assert kusunda["Clade"] == "Other"
    assert kusunda["Quality"] == "B"
    dialects = dicts(ROOT / "cldf/dialects.csv")
    assert not any(row["Language_ID"] == "Kusunda" for row in dialects)


def test_compiled_rows_survive_as_unlinked_nodes_with_source_layers_separated():
    compiled = [
        row for row in dicts(ROOT / "cldf/forms.csv")
        if SOURCE_KEY in row["Source"]
    ]
    if not compiled:
        pytest.skip("Run make all to exercise compiled Kusunda assertions")

    assert len(compiled) == 662
    assert {row["Language_ID"] for row in compiled} == {"Kusunda"}
    from reviewed_graph_policy import assert_reviewed_source_graph
    assert_reviewed_source_graph(compiled)
    assert all(not row["Cognateset"] for row in compiled)
    assert all(row["Original"] == row["Phonemic"] for row in compiled)
    assert sum(row["Form"].startswith("*") for row in compiled) == 211
    assert all(
        row["Form"].startswith("*") == ("ground-form reconstruction" in row["Source"])
        for row in compiled
    )

    source_keys = dicts(ROOT / "cldf/form-source-keys.csv")
    kusunda_keys = [row for row in source_keys if row["Source_Key"].startswith((
        "ProtoKusunda-", "KusundaGM-", "KusundaK-"
    ))]
    assert len(kusunda_keys) == 662
    assert {r['Source_Key'] for r in kusunda_keys} == {r[10] for r in rows(INSTALLED)}
    aliases = {r['Legacy_ID']:r['Form_ID'] for r in dicts(ROOT / 'cldf/form-id-aliases.csv')}
    assert {aliases[r['Legacy_ID']] for r in kusunda_keys} == {r['ID'] for r in compiled}


def test_reference_metadata_is_complete_after_build():
    references_path = ROOT / "cldf/references.csv"
    if not references_path.exists():
        pytest.skip("Run make all to exercise compiled reference assertions")
    references = {row["ID"]: row for row in dicts(references_path)}
    if SOURCE_KEY not in references:
        pytest.skip("Run make all to refresh compiled references")
    ref = references[SOURCE_KEY]
    assert "Aaley" in ref["Source"] and "Bodt" in ref["Source"]
    assert ref["Progress"]
    assert "20260901-aaley-bodt-kusunda.csv" in ref["Provenance"]
    assert ref["OCR"] == "No"
    assert ref["Etymology_Provenance"] == "none"
