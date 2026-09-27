"""Focused source checks for Bailey's complete Bilaspuri vocabulary."""

import csv
import importlib.util
import io
import json
import hashlib
import unicodedata
import sys
from collections import Counter
from pathlib import Path

from segments.tokenizer import Tokenizer


DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / "data/other/forms/raw_data/bailey_bilaspuri_1920"
CSV = DATA / "data/other/forms/20260925-bailey-bilaspuri.csv"
PROFILE = DATA / "conversion/bailey-bilaspuri-1920.txt"
spec = importlib.util.spec_from_file_location("bailey_bilaspuri_1920", PACKAGE / "import_source.py")
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
    assert rows == installed() and audit == audited()
    assert len(audit) == 237 and len(rows) == 288
    assert Counter(a["status"] for a in audit) == {
        "ingested": 237,
    }
    assert [(a["printed_page"], a["column"], a["vocabulary_item"]) for a in audit] == [
        (245, "left", item) for item in range(1, 34)
    ] + [
        (page, column, item)
        for page, column, count in (
            (245, "right", 35), (246, "left", 36), (246, "right", 37),
            (247, "left", 36), (247, "right", 37),
            (248, "left", 13), (248, "right", 10),
        )
        for item in range(1, count + 1)
    ]
    assert all(a["scan_page"] == a["printed_page"] + 26 for a in audit)
    assert len({r[10] for r in rows}) == 288


def test_legacy_identity_and_literal_corrections():
    rows = {r[10]: r for r in installed()}
    legacy = json.loads((PACKAGE / "legacy-entry-keys.json").read_text())
    assert len(legacy) == 114 and set(legacy) <= set(rows)
    for key in ("p245:left:item:29", "p247:left:item:14", "p247:left:item:34:answer2"):
        assert rows["bailey1920bilaspuri:" + key][2] == "bōlṇā"
    assert rows["bailey1920bilaspuri:p245:right:item:2"][2] == "gău̇"
    assert rows["bailey1920bilaspuri:p245:right:item:2:answer2"][2] == "gāẽ"
    assert rows["bailey1920bilaspuri:p245:left:item:30"][2] == "ū̃ṭ"
    assert rows["bailey1920bilaspuri:p246:right:item:12"][2] == "mās̲h̲"


def test_expanded_grammar_alignment():
    rows = {r[10]: r for r in installed()}
    def answers(key):
        return [(r[2], r[3], r[14]) for k, r in rows.items()
                if k == "bailey1920bilaspuri:" + key or k.startswith("bailey1920bilaspuri:" + key + ":answer")]
    assert answers("p245:right:item:11") == [("pīṇā", "drink", ""), ("pĭḷāṇā", "give to drink", "caus")]
    assert answers("p246:left:item:10") == [("cārnā", "graze", "tr"), ("cŭgāṇā", "graze", "tr"), ("cŭgṇā", "graze", "intr")]
    assert answers("p245:left:item:31") == [("bĭllā", "cat", ""), ("bĭllī", "cat", "f")]
    assert answers("p248:left:item:13") == [("kŭn", "who", "interr"), ("jō", "who", "relative")]


def test_fresh_independent_audit_pins_installed_readings():
    report = json.loads((PACKAGE / "independent-literal-audit-20260926-pass2.json").read_text())
    assert report["sample_size"] == len(report["entries"]) == 20
    assert report["material_errors"] == 0
    assert hashlib.sha256((PACKAGE / "full-transcription.tsv").read_bytes()).hexdigest() == report["input_sha256"]["full-transcription-staged.tsv"]
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == report["input_sha256"]["literal-staged.csv"]
    assert Counter(e["page"] for e in report["entries"]) == {str(p): 5 for p in range(245, 249)}


def test_metadata_profile_and_parse():
    import make_cldf
    import profile_policy
    import source_meta
    from tags import GRAMMATICAL_TAGS, GENDER_TAGS

    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert len(row) == 15 and row[0] == "bil"
        assert row[1] == row[4] == row[5] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
        assert unicodedata.normalize("NFC", tokenizer(row[2], column="IPA").replace(" ", "").replace("#", "")) == row[2].replace("w", "v").replace("ṅ", "ŋ").replace(" ", "")
        assert set(row[14].split()) <= set(GRAMMATICAL_TAGS) | set(GENDER_TAGS)
    assert source_meta.SourceMeta().transcription("bailey1920bilaspuri", CSV, "bil")[0] == "bailey-bilaspuri-1920"
    assert "bailey-bilaspuri-1920" not in profile_policy.audit({})
    assert "@book{bailey1920bilaspuri," in (DATA / "cldf/sources.bib").read_text()
    assert "bil" in {r[0] for r in csv.reader((DATA / "cldf/languages.csv").open())}
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-bailey-bilaspuri")
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 288
    raw = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(raw)
    assert all(r.old_form == raw[r.entry_key][2] for r in parsed)
