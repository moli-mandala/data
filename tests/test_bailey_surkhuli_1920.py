"""Focused checks for Bailey's complete Surkhuli source, without a DB build."""

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
PACKAGE = DATA / "data/other/forms/raw_data/bailey_surkhuli_1920"
CSV = DATA / "data/other/forms/20260925-bailey-surkhuli.csv"
PROFILE = DATA / "conversion/bailey-surkhuli-1920.txt"
spec = importlib.util.spec_from_file_location("bailey_surkhuli_1920", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_full_glossary_reconciliation_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed()
    assert audit == audited()
    assert len(audit) == 529 and len(rows) == 605
    assert rows[:267] == list(csv.reader((PACKAGE / "legacy-before-whole-recovery.csv").open()))
    assert Counter(a["status"] for a in audit[224:]) == {"target":301, "morphology_with_context":3, "control":1}
    audit = audit[:224]
    assert Counter(a["status"] for a in audit) == {"ingested": 223, "cross_reference": 1}
    assert [(a["printed_page"],a["vocabulary_item"]) for a in audit] == [(p,i) for p,n in [(155,69),(156,70),(157,72),(158,13)] for i in range(1,n+1)]
    assert {(a["printed_page"], a["scan_page"]) for a in audit} == {(p,p+26) for p in range(155,159)}
    assert len({r[10] for r in rows}) == 605


def test_source_answer_keys_and_exclusions():
    audit = {a["vocabulary_item"]: a for a in audited() if a["printed_page"] == 155}
    rows = {r[10]: r for r in installed()}
    for item in (4, 12, 64):
        assert len(audit[item]["entry_keys"]) == 2
        assert rows[f"bailey1920surkhuli:p155:item:{item}:answer2"][11] == ""
    assert audit[25]["status"] == "ingested"
    assert audit[46]["status"] == "ingested"
    assert rows["bailey1920surkhuli:p155:item:1"][7] == "bailey1920surkhuli[p. 155, vocabulary item 1]"


def test_legacy_keys_grammar_and_source_notes():
    rows = {r[10]:r for r in installed()}
    legacy = json.loads((PACKAGE / "legacy-entry-keys.json").read_text())
    assert len(legacy) == 29 and set(legacy) <= set(rows)
    for item, form in {6:"nĭkāmmau",11:"mănzā",15:"tsīṛū",18:"kătāb"}.items():
        assert rows[f"bailey1920surkhuli:p155:item:{item}"][2] == form
    assert "very long" in rows["bailey1920surkhuli:p155:item:49"][6]
    assert "very long" in rows["bailey1920surkhuli:p157:item:39"][6]
    assert "very long" in rows["bailey1920surkhuli:p157:item:52"][6]
    assert "accent on first syllable" in rows["bailey1920surkhuli:p156:item:50"][6]
    assert rows["bailey1920surkhuli:p156:item:1:answer2"][14] == "f"
    assert rows["bailey1920surkhuli:p156:item:44:answer5"][14] == "interr"
    assert rows["bailey1920surkhuli:p156:item:44:answer7"][14] == "relative"
    assert rows["bailey1920surkhuli:p156:item:44:answer9"][14] == "adv"
    assert rows["bailey1920surkhuli:p157:item:63:answer2"][14] == "f"
    assert rows["bailey1920surkhuli:p158:item:8"][14] == "instr"


def test_profile_policy_language_and_bibliography():
    import profile_policy
    import source_meta
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS

    keys = {r[10] for r in installed()}
    assert sum(bool(r[11]) for r in installed()) == 27
    assert all(not r[11] or r[11] in keys for r in installed())
    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert len(row) == 15 and row[0] == "surkh"
        assert row[1] == row[4] == row[5] == row[8] == row[9] == row[12] == row[13] == ""
        assert row[2] and "�" not in row[2]
        original = unicodedata.normalize("NFC", row[2])
        assert unicodedata.normalize("NFC", tokenizer(original, column="IPA").replace(" ", "").replace("#", " ")) == original.replace("w", "v").replace("ṅ", "ŋ").replace(".", "")
        assert set(row[14].split()) <= set(GRAMMATICAL_TAGS) | set(GENDER_TAGS) | {"uncertain"}
    assert source_meta.SourceMeta().transcription("bailey1920surkhuli", CSV, "surkh")[0] == "bailey-surkhuli-1920"
    rule = source_meta.SourceMeta().transcription_rule("bailey1920surkhuli", CSV, "surkh")
    assert rule["preserve_hyphens"] is True
    assert "bailey-surkhuli-1920" not in profile_policy.audit({})
    import yaml
    meta = yaml.safe_load(CSV.with_suffix(".yaml").read_text())
    assert meta["defaults"]["identity"] == {"legacy_ids":"stem", "append_order":49}
    assert meta["sources"]["bailey1920surkhuli"]["reference"]["ocr"] is True
    assert meta["sources"]["bailey1920surkhuli"]["forms"]["split_alternates"] is False
    assert "@book{bailey1920surkhuli," in (DATA / "cldf/sources.bib").read_text()
    languages = {r[0] for r in csv.reader((DATA / "cldf/languages.csv").open())}
    assert "surkh" in languages


def test_cldf_parse_preserves_originals():
    import make_cldf

    errors = io.StringIO()
    rows, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-bailey-surkhuli")
    raw = {r[10]: r for r in installed()}
    assert not errors.getvalue()
    assert len(rows) == stats["converted"] == 605
    assert {r.entry_key for r in rows} == set(raw)
    for row in rows:
        r = raw[row.entry_key]
        assert (row.old_form,row.notes,row.source,row.tags) == (r[2],r[6],r[7],r[14])
        assert row.form == r[2].replace("w","v").replace("ṅ","ŋ").replace(".","")
    old_errors = io.StringIO()
    old, _ = make_cldf.parse_file(str(PACKAGE / "legacy-before-whole-recovery.csv"), old_errors, name="20260925-bailey-surkhuli")
    assert not old_errors.getvalue() and len(old) == 267
    current_ids = {r.entry_key:r.id for r in rows}
    assert all(current_ids[r.entry_key] == r.id for r in old)


def test_return_punctuation_preserved_as_evidence_not_display():
    rows = {r[10]:r for r in installed()}
    row = rows["bailey1920surkhuli:p157:item:2"]
    assert row[2] == "ōru. ăs̲h̲ṇo" and row[14] == "uncertain"
    display = Tokenizer(str(PROFILE))(row[2], column="IPA").replace(" ", "").replace("#", " ")
    assert display == "ōru ăs̲h̲ṇo"
    assert "word separation" in row[6]


def test_independent_audit_and_narrow_normalization_addendum():
    import hashlib

    review = json.loads((PACKAGE / "independent-literal-audit-20260926-pass1.json").read_text())
    assert review["sample_size"] == 20 and review["material_errors"] == 0
    assert Counter(row["page"] for row in review["entries"]) == {str(p): 5 for p in range(155, 159)}
    addendum = json.loads((PACKAGE / "return-normalization-addendum-20260926.json").read_text())
    assert addendum["prior_sample_size"] == 20
    assert addendum["prior_sample_readings_glosses_tags_unchanged"] is True
    assert addendum["period_scope_cells"] == [["157", "2"]]
    historical = {
        "data/other/forms/20260925-bailey-surkhuli.csv": PACKAGE / "legacy-before-whole-recovery.csv",
        "conversion/bailey-surkhuli-1920.txt": PACKAGE / "literal-profile-staged.txt",
        "data/other/forms/raw_data/bailey_surkhuli_1920/audit.jsonl": PACKAGE / "legacy-before-whole-recovery-audit.jsonl",
    }
    for path, expected in addendum["sha256"].items():
        assert hashlib.sha256(historical.get(path, DATA / path).read_bytes()).hexdigest() == expected
    approval = json.loads((PACKAGE / "root-morphology-continuity-approval-20260926.json").read_text())
    for filename, canonical in [("whole-proposed.csv",CSV),("whole-proposed-audit.jsonl",PACKAGE/"audit.jsonl"),("whole-proposed-profile.txt",PROFILE)]:
        assert hashlib.sha256(canonical.read_bytes()).hexdigest() == approval["sha256"][filename]
        assert canonical.read_bytes() == (PACKAGE / filename).read_bytes()
