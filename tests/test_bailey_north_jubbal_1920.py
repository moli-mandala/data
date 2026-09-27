"""Focused source-input checks for Bailey's complete North Jubbal glossary."""

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
PACKAGE = DATA / "data/other/forms/raw_data/bailey_north_jubbal_1920"
CSV = DATA / "data/other/forms/20260925-bailey-north-jubbal.csv"
PROFILE = DATA / "conversion/bailey-north-jubbal-1920.txt"
spec = importlib.util.spec_from_file_location("bailey_north_jubbal_1920", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_exact_page_scope_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed()
    assert audit == audited()
    assert len(audit) == 243 and len(rows) == 330
    assert Counter(a["status"] for a in audit) == {
        "ingested": 242,
        "cross_reference": 1,
    }
    assert [(a["printed_page"],a["vocabulary_item"]) for a in audit] == [(p,i) for p,n in [(185,71),(186,68),(187,71),(188,33)] for i in range(1,n+1)]
    assert {(a["printed_page"], a["scan_page"]) for a in audit} == {(p,p+26) for p in range(185,189)}
    assert len({r[10] for r in rows}) == 330


def test_answer_keys_and_printed_mark_decisions():
    audit = {a["vocabulary_item"]: a for a in audited()[:71]}
    rows = {r[10]: r for r in installed()}
    assert audit[1]["status"] == "cross_reference"
    for item in (20, 22, 56, 65):
        assert len(audit[item]["entry_keys"]) == 2
        assert rows[f"bailey1920northjubbal:p185:item:{item}:answer2"][11] == ""
    assert rows["bailey1920northjubbal:p185:item:13"][2] == "pătshu"
    assert rows["bailey1920northjubbal:p185:item:17"][2] == "bŏṛo"
    assert audit[25]["status"] == "ingested"
    assert audit[30]["status"] == "ingested"


def test_literal_recovery_and_preserved_keys():
    rows = {r[10]: r for r in installed()}
    legacy = json.loads((PACKAGE / "legacy-entry-keys.json").read_text())
    assert len(legacy) == 98 and set(legacy) <= set(rows)
    for key, form in {
        "p185:item:5":"tshŏū̃", "p185:item:12":"gŏū̃", "p185:item:28":"budno",
        "p185:item:36":"gaīḷā", "p185:item:37":"dī‘ī", "p185:item:70":"tshāṅṭi",
        "p186:item:5":"măṇḍăḷ", "p186:item:40":"bēlṛī", "p187:item:5":"tshāṛno",
        "p187:item:21":"bōlṇo", "p187:item:42":"bōlṇo", "p187:item:61":"tū",
    }.items():
        assert rows["bailey1920northjubbal:" + key][2] == form
    assert rows["bailey1920northjubbal:p186:item:51:answer3"][3] == "so much"
    assert rows["bailey1920northjubbal:p186:item:51:answer4"][14] == "interr"
    assert rows["bailey1920northjubbal:p186:item:51:answer5"][14] == "relative"


def test_profile_language_metadata_and_bibliography():
    import profile_policy
    import source_meta
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS

    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert len(row) == 15 and row[0] == "Barari"
        assert row[1] == row[4] == row[5] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
        assert row[2] and "�" not in row[2]
        original = unicodedata.normalize("NFC", row[2])
        assert unicodedata.normalize("NFC", tokenizer(original, column="IPA").replace(" ", "").replace("#", " ")) == original.replace("w", "v").replace("ṅ", "ŋ")
        assert set(row[14].split()) <= set(GRAMMATICAL_TAGS) | set(GENDER_TAGS)
    assert source_meta.SourceMeta().transcription("bailey1920northjubbal", CSV, "Barari")[0] == "bailey-north-jubbal-1920"
    assert "bailey-north-jubbal-1920" not in profile_policy.audit({})
    assert "@book{bailey1920northjubbal," in (DATA / "cldf/sources.bib").read_text()
    languages = {r[0] for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert "Barari" in languages


def test_cldf_parse_preserves_originals():
    import make_cldf

    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-bailey-north-jubbal")
    raw = {r[10]: r for r in installed()}
    assert not errors.getvalue()
    assert len(rows) == stats["converted"] == 330
    assert {r.entry_key for r in rows} == set(raw)
    assert all(r.old_form == raw[r.entry_key][2] for r in rows)


def test_pilot_corrections_and_answer_glosses():
    rows = {r[10]:r for r in installed()}
    for item, spelling in {17:"bŏṛo",21:"tshōṭā",43:"gāṛno",47:"pinni",54:"pŏṛno",56:"bābbā"}.items():
        assert rows[f"bailey1920northjubbal:p185:item:{item}"][2] == spelling
    assert rows["bailey1920northjubbal:p188:item:28:answer3"][2:4] == ["dā","with"]
    assert rows["bailey1920northjubbal:p188:item:16"][14] == "adv"
    assert rows["bailey1920northjubbal:p188:item:17"][14] == "noun"



def test_fresh_full_literal_audit_pins_install():
    import hashlib
    report = json.loads((PACKAGE / "independent-literal-audit-20260926-pass2.json").read_text())
    assert report["sample_size"] == len(report["entries"]) == 20
    assert report["material_errors"] == 0
    assert Counter(e["page"] for e in report["entries"]) == {str(p):5 for p in range(185,189)}
    assert hashlib.sha256((PACKAGE / "full-transcription.tsv").read_bytes()).hexdigest() == report["hashes"]["full-transcription-staged.tsv"]
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == report["hashes"]["literal-staged.csv"]
