"""Focused source-input checks for Bailey's complete Baghi glossary comparison."""

import csv
import importlib.util
import io
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / "data/other/forms/raw_data/bailey_baghi_1920"
CSV = DATA / "data/other/forms/20260925-bailey-baghi.csv"
PROFILE = DATA / "conversion/bailey-baghi-1920.txt"
spec = importlib.util.spec_from_file_location("bailey_baghi_1920", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_page_reconciliation_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed()
    assert audit == audited()
    assert len(audit) == 245 and len(rows) == 296
    assert Counter(a["status"] for a in audit) == {
        "ingested": 240,
        "cross_reference": 2, "hold_no_baghi": 3,
    }
    assert [(a["printed_page"],a["vocabulary_item"]) for a in audit] == [(p,i) for p,n in [(144,59),(145,66),(146,63),(147,57)] for i in range(1,n+1)]
    assert {(a["printed_page"], a["scan_page"]) for a in audit} == {(p,p+26) for p in range(144,148)}
    assert len({r[10] for r in rows}) == 296


def test_column_provenance_and_exclusions():
    audit = {a["vocabulary_item"]: a for a in audited()[:59]}
    rows = {r[10]: r for r in installed()}
    assert all("Rampur" in a["control_column"] for a in audit.values())
    assert audit[1]["status"] == audit[35]["status"] == "cross_reference"
    assert audit[52]["status"] == "hold_no_baghi"
    assert audit[25]["status"] == "ingested"
    assert rows["bailey1920baghi:p144:item:3"][7] == "bailey1920baghi[p. 144, vocabulary item 3, Baghi column]"
    assert all(r[11] == "" for r in rows.values())


def test_legacy_keys_and_literal_error_classes():
    rows = {r[10]: r for r in installed()}
    legacy = json.loads((PACKAGE / "legacy-entry-keys.json").read_text())
    assert len(legacy) == 96 and set(legacy) <= set(rows)
    assert rows["bailey1920baghi:p144:item:46:answer2"][2:4] == ["pĩnēṇo", "cause to drink"]
    assert rows["bailey1920baghi:p144:item:46:answer2"][14] == "caus"
    assert rows["bailey1920baghi:p144:item:48:answer2"][14] == "caus"
    assert rows["bailey1920baghi:p146:item:30"][2] == "dăryaio"
    assert rows["bailey1920baghi:p145:item:49"][2] == "s̲h̲īkṇo"
    assert rows["bailey1920baghi:p145:item:45:answer4"][14] == "relative"
    assert rows["bailey1920baghi:p145:item:66:answer5"][14] == "interr"
    assert rows["bailey1920baghi:p145:item:66:answer6"][14] == "relative"


def test_profile_language_metadata_and_bibliography():
    import profile_policy
    import source_meta
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS

    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert len(row) == 15 and row[0] == "ba"
        assert row[1] == row[4] == row[5] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
        assert row[2] and "�" not in row[2]
        original = unicodedata.normalize("NFC", row[2])
        assert unicodedata.normalize("NFC", tokenizer(original, column="IPA").replace(" ", "").replace("#", "")) == original.replace("ṅ", "ŋ").replace("w", "v").replace(" ", "")
        assert set(row[14].split()) <= set(GRAMMATICAL_TAGS) | set(GENDER_TAGS)
    assert source_meta.SourceMeta().transcription("bailey1920baghi", CSV, "ba")[0] == "bailey-baghi-1920"
    assert "bailey-baghi-1920" not in profile_policy.audit({})
    assert "@book{bailey1920baghi," in (DATA / "cldf/sources.bib").read_text()
    languages = {r[0] for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert "ba" in languages


def test_cldf_parse_preserves_originals():
    import make_cldf

    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-bailey-baghi")
    raw = {r[10]: r for r in installed()}
    assert not errors.getvalue()
    assert len(rows) == stats["converted"] == 296
    assert {r.entry_key for r in rows} == set(raw)
    assert all(r.old_form == raw[r.entry_key][2] for r in rows)


def test_marked_consonants_and_multi_answer_glosses():
    rows = {r[10]:r for r in installed()}
    assert rows["bailey1920baghi:p145:item:19"][2] == "ḍūṇo"
    assert rows["bailey1920baghi:p145:item:26"][2] == "mūṇḍ"
    assert rows["bailey1920baghi:p147:item:12:answer2"][2:4] == ["cījjo","third"]
    assert rows["bailey1920baghi:p147:item:30"][14] == "adj"
    audit = {x["source_cell_key"]:x for x in audited()}
    assert len(audit["bailey1920baghi:p145:item:66"]["entry_keys"]) == 6


def test_corrected_pilot_breves_and_vowel_length():
    rows = {r[10]:r for r in installed()}
    for item, expected in {20:"kătāb",22:"rōṭṭi",24:"bāĭh",31:"ătshau"}.items():
        assert rows[f"bailey1920baghi:p144:item:{item}"][2] == expected


def test_fresh_independent_sourcewide_audit():
    import hashlib
    report = json.loads((PACKAGE / "independent-literal-audit-20260926-pass2.json").read_text())
    assert report["sample_size"] == 20 and report["material_errors"] == 0
    assert Counter(int(x["page"]) for x in report["entries"]) == {144:5,145:5,146:5,147:5}
    assert hashlib.sha256((PACKAGE / "full-transcription.tsv").read_bytes()).hexdigest() == report["hashes"]["full-transcription-staged.tsv"]
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == report["hashes"]["literal-staged.csv"]
