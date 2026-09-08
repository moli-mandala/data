"""Regression tests for Aaley's complete 2021 *Kusunda Gipan* glossary."""

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
PACKAGE = ROOT / "data/other/forms/raw_data/aaley_kusunda_gipan_2021"
IMPORTER = PACKAGE / "import_gipan.py"
INSTALLED = ROOT / "data/other/forms/20260901-aaley-kusunda-gipan.csv"
AUDIT = PACKAGE / "20260901-aaley-kusunda-gipan-audit.csv"
SOURCE_KEY = "aaley2021kusundagipan"

SPEC = importlib.util.spec_from_file_location("aaley_kusunda_gipan", IMPORTER)
assert SPEC and SPEC.loader
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


def dicts(path, delimiter=","):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_pinned_legacy_font_ledger_and_complete_table_census():
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    snapshot = PACKAGE / manifest["snapshot"]["file"]
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == manifest["snapshot"]["sha256"]
    raw = source.read_snapshot()
    assert len(raw) == 160
    assert Counter(row["PDF_Page"] for row in raw) == Counter({"54": 70, "55": 74, "56": 16})
    assert {row["Form_Devanagari"] for row in raw if "/" in row["Form_Devanagari"]} == {"घै/गहि", "मुङ/मोङ"}


def test_variants_duplicate_and_audit_are_explicit():
    installed = rows(INSTALLED)
    audit = dicts(AUDIT)
    assert len(installed) == 161 and len(audit) == 162
    assert Counter(row["Status"] for row in audit) == Counter(ingested=161, excluded=1)
    duplicate = [row for row in audit if row["Status"] == "excluded"]
    assert len(duplicate) == 1
    assert duplicate[0]["Form_Devanagari"] == "पाङजाङ" and duplicate[0]["Gloss"] == "five"
    assert sum(bool(row[11]) for row in installed) == 2
    assert len([row for row in audit if row["Review_State"] == "verified-against-render"]) == 20


def test_native_spelling_romanization_and_translation_are_preserved():
    by_key = {row[10]: row for row in rows(INSTALLED)}
    mango = by_key[f"{SOURCE_KEY}:p48:c1:r04:v01"]
    assert (mango[2], mango[3], mango[4]) == ("əmbyak", "mango", "अम्ब्याक")
    assert "Nepali gloss: आँप" in mango[6]
    wound = by_key[f"{SOURCE_KEY}:p49:c1:r02:v02"]
    assert (wound[2], wound[4]) == ("gəhi", "गहि")
    assert wound[11].endswith(":v01")
    king = by_key[f"{SOURCE_KEY}:p49:c2:r23:v02"]
    assert (king[2], king[3], king[4]) == ("moṅ", "king", "मोङ")
    assert all(unicodedata.normalize("NFC", value) == value for row in by_key.values() for value in row)


def test_profile_covers_every_romanized_form():
    tokenizer = Tokenizer(str(ROOT / "conversion/kusunda-gipan.txt"))
    converted = []
    for row in rows(INSTALLED):
        display = unicodedata.normalize("NFC", tokenizer(row[2], column="IPA").replace(" ", "").replace("#", " "))
        converted.append((row[2], display))
    assert not [(raw, display) for raw, display in converted if "�" in display]
    assert unicodedata.normalize("NFC", tokenizer("khaṅgu", column="IPA").replace(" ", "")) == "kʰaŋgu"


def test_compiled_rows_and_reference_metadata():
    forms_path = ROOT / "cldf/forms.csv"
    refs_path = ROOT / "cldf/references.csv"
    if not forms_path.exists() or not refs_path.exists():
        pytest.skip("Run make all to exercise compiled assertions")
    compiled = [row for row in dicts(forms_path) if SOURCE_KEY in row["Source"]]
    refs = {row["ID"]: row for row in dicts(refs_path)}
    if not compiled or SOURCE_KEY not in refs:
        pytest.skip("Run make all to refresh compiled assertions")
    assert len(compiled) == 161
    assert {row["Language_ID"] for row in compiled} == {"Kusunda"}
    assert all(row["Native"] for row in compiled)
    identities = {
        row["Source_Key"]: row["Form_ID"] for row in dicts(ROOT / "data/form-identities.csv")
        if row["Source_Key"].startswith(f"{SOURCE_KEY}:")
    }
    assert len(identities) == 161
    variant_edges = {
        (row["Child_ID"], row["Parent_ID"])
        for row in dicts(ROOT / "cldf/edges.csv") if row["Kind"] == "variant"
    }
    for row in rows(INSTALLED):
        if row[11]:
            assert (identities[row[10]], identities[row[11]]) in variant_edges
    assert refs[SOURCE_KEY]["OCR"] == "No"
    assert "20260901-aaley-kusunda-gipan.csv" in refs[SOURCE_KEY]["Provenance"]
