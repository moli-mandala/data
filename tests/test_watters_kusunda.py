"""Regression tests for Watters's complete 2006 Kusunda vocabulary."""

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
PACKAGE = ROOT / "data/other/forms/raw_data/watters_kusunda_2006"
IMPORTER = PACKAGE / "import_watters.py"
INSTALLED = ROOT / "data/other/forms/20260901-watters-kusunda.csv"
AUDIT = PACKAGE / "20260901-watters-kusunda-audit.csv"
SOURCE_KEY = "watters2006kusunda"

SPEC = importlib.util.spec_from_file_location("watters_kusunda", IMPORTER)
assert SPEC and SPEC.loader
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


def dicts(path, delimiter=","):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_pinned_unicode_control_and_complete_source_census():
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    snapshot = PACKAGE / manifest["unicode_control"]["file"]
    assert manifest["unicode_control"]["revision"] == "84645357"
    assert manifest["unicode_control"]["license"] == "CC-BY-SA-4.0"
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == manifest["unicode_control"]["sha256"]
    records = source.parse_records()
    assert len(records) == 877
    assert Counter(row["pos"] for row in records) == Counter({
        "n.": 401, "vt.": 211, "vi.": 104, "adj.": 53, "adv.": 40,
        "pp.": 18, "pron.": 18, "num.": 10, "interrog.": 6, "aff.": 3,
        "dem.": 3, "v.": 3, "loc.": 2, "suff.": 1, "": 1,
        "conj.": 1, "v., adj.": 1, "greet.": 1,
    })
    assert sum(left + right for _, left, right in source.PAGE_COLUMN_COUNTS) == 877


def test_all_attested_forms_are_split_and_linked_without_loss():
    installed = rows(INSTALLED)
    audit = dicts(AUDIT)
    assert len(installed) == len(audit) == 1387
    assert {row["Status"] for row in audit} == {"ingested"}
    assert all(len(row) == 15 for row in installed)
    assert len({row[10] for row in installed}) == len(installed)
    assert {row[0] for row in installed} == {"Kusunda"}
    assert all(row[7].startswith(f"{SOURCE_KEY}[") for row in installed)
    assert all(unicodedata.normalize("NFC", value) == value for row in installed for value in row)
    assert sum(bool(row[11]) for row in installed) == 510
    assert sum("sound-variant" in row[14].split() for row in installed) == 118
    assert len([row for row in audit if row["Review_State"] == "verified-against-render"]) == 20


def test_representative_paradigms_variants_loans_and_locations():
    by_key = {row[10]: row for row in rows(INSTALLED)}
    garlic = by_key[f"{SOURCE_KEY}:p139:c1:i014:v02"]
    assert garlic[2] == "əraχ" and garlic[11].endswith(":v01")
    assert {"noun", "alternate", "sound-variant"} <= set(garlic[14].split())
    beg_prohibitive = by_key[f"{SOURCE_KEY}:p139:c2:i007:v05"]
    assert beg_prohibitive[2] == "ai-yin"
    assert {"verb", "tr", "impv", "neg", "alternate"} <= set(beg_prohibitive[14].split())
    potato = by_key[f"{SOURCE_KEY}:p139:c2:i012:v01"]
    assert potato[2] == "alu"
    assert {"noun", "loanword", "loan:Nepali"} <= set(potato[14].split())
    assert by_key[f"{SOURCE_KEY}:p144:c1:i032:v01"][3] == "'I am hungry' (lit. hunger is to me)"


def test_profile_covers_every_installed_form():
    tokenizer = Tokenizer(str(ROOT / "conversion/kusunda-watters.txt"))
    converted = []
    for row in rows(INSTALLED):
        display = unicodedata.normalize("NFC", tokenizer(row[2], column="IPA").replace(" ", "").replace("#", " "))
        converted.append((row[2], display))
    assert not [(raw, display) for raw, display in converted if "�" in display]
    assert unicodedata.normalize("NFC", tokenizer("əraχ", column="IPA").replace(" ", "")) == "ərax"


def test_compiled_rows_and_reference_metadata():
    forms_path = ROOT / "cldf/forms.csv"
    refs_path = ROOT / "cldf/references.csv"
    if not forms_path.exists() or not refs_path.exists():
        pytest.skip("Run make all to exercise compiled assertions")
    compiled = [row for row in dicts(forms_path) if SOURCE_KEY in row["Source"]]
    refs = {row["ID"]: row for row in dicts(refs_path)}
    if not compiled or SOURCE_KEY not in refs:
        pytest.skip("Run make all to refresh compiled assertions")
    assert len(compiled) == 1387
    assert {row["Language_ID"] for row in compiled} == {"Kusunda"}
    # Source variants plus later accepted loans determine status, not the pre-review count.
    identities = {
        row["Source_Key"]: row["Form_ID"] for row in dicts(ROOT / "data/form-identities.csv")
        if row["Source_Key"].startswith(f"{SOURCE_KEY}:")
    }
    assert len(identities) == 1387
    assert set(identities) == {r[10] for r in rows(INSTALLED)}
    assert set(identities.values()) == {r['ID'] for r in compiled}
    variant_edges = {
        (row["Child_ID"], row["Parent_ID"])
        for row in dicts(ROOT / "cldf/edges.csv") if row["Kind"] == "variant"
    }
    for row in rows(INSTALLED):
        if row[11]:
            assert (identities[row[10]], identities[row[11]]) in variant_edges
    from reviewed_graph_policy import assert_reviewed_source_graph
    source_edges = {(identities[r[10]],identities[r[11]],'variant','1','') for r in rows(INSTALLED) if r[11]}
    assert_reviewed_source_graph(compiled, source_edges)
    assert refs[SOURCE_KEY]["OCR"] == "No"
    assert "20260901-watters-kusunda.csv" in refs[SOURCE_KEY]["Provenance"]
