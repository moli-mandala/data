"""Focused source-input checks for Bailey's complete Rohru glossary."""

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
PACKAGE = DATA / "data/other/forms/raw_data/bailey_rohru_1920"
CSV = DATA / "data/other/forms/20260925-bailey-rohru.csv"
PROFILE = DATA / "conversion/bailey-rohru-1920.txt"
spec = importlib.util.spec_from_file_location("bailey_rohru_1920", PACKAGE / "import_source.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def installed():
    with CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def audited():
    return [json.loads(line) for line in (PACKAGE / "audit.jsonl").read_text().splitlines()]


def test_complete_column_and_regeneration():
    rows, audit = source.generate()
    assert rows == installed() and audit == audited()
    assert len(audit) == 255 and len(rows) == 309
    assert Counter(a["status"] for a in audit) == {
        "ingested": 250, "hold_typography": 2, "exclude_crossref": 3
    }
    assert [(a["printed_page"], a["column"], a["editorial_line_number"]) for a in audit] == [
        (p,c,i) for p,c,n in source.SCOPE for i in range(1,n+1)]
    assert {(a["printed_page"], a["scan_page"], a["column"]) for a in audit} == {(p,p+26,c) for p,c,n in source.SCOPE}
    assert len({r[10] for r in rows}) == 309


def test_mapping_separate_answers_and_explicit_grammar():
    rows = {r[10]: r for r in installed()}
    key = "bailey1920rohru:p127:left:line:6"
    assert all(r[0] == "roh" and r[11] == "" for r in rows.values())
    assert rows[key][2:4] == ["kōi", "anyone"]
    assert rows[key + ":answer2"][2:4] == ["kicch", "anything"]
    assert rows["bailey1920rohru:p127:left:line:9"][3:] [-1] == "noun"
    assert rows["bailey1920rohru:p127:left:line:10"][2] == "patsho"
    assert rows["bailey1920rohru:p127:left:line:17"][2] == "patshe"
    assert audited()[2]["status"] == "exclude_crossref"


def test_recovered_diacritics_and_scoped_senses():
    rows = {r[10]: r for r in installed()}
    def row(page, column, line, answer=""):
        return rows[f"bailey1920rohru:p{page}:{column}:line:{line}{answer}"]
    assert row(128,"right",29,":answer2")[2] == "bŏhri"
    assert row(130,"left",24)[2] == "bōhri"
    assert row(130,"left",22)[2] == "hūbi"
    assert row(130,"right",9)[2] == "gīū̃h"
    assert row(128,"right",5)[2] == "s̲h̲aiḷṭo"
    assert row(127,"right",16,":answer2")[3] == "cause to drink"
    assert row(127,"right",16,":answer2")[14] == "caus"
    assert row(128,"left",20)[14] == "verb intr"
    assert row(128,"left",20,":answer2")[14] == "verb tr"
    assert not any("reading withheld" in a["printed_form_review"] for a in audited())
    holds = [a for a in audited() if a["status"] == "hold_typography"]
    assert {(a["printed_page"],a["column"],a["editorial_line_number"]) for a in holds} == {(129,"right",25),(130,"left",25)}
    assert all("stack" in a["reason"] for a in holds)


def test_profile_metadata_and_parse():
    import make_cldf
    import profile_policy
    import source_meta

    tokenizer = Tokenizer(str(PROFILE))
    for row in installed():
        assert len(row) == 15
        assert row[1] == row[4] == row[5] == row[6] == row[8] == row[9] == row[11] == row[12] == row[13] == ""
        original = unicodedata.normalize("NFC", row[2])
        assert tokenizer(original, column="IPA").replace(" ", "").replace("#", " ") == original
        parts = row[10].split(":")
        assert row[7] == f"bailey1920rohru[p. {parts[1][1:]}, {parts[2]} column, line {parts[4]}]"
    assert source_meta.SourceMeta().transcription("bailey1920rohru", CSV, "roh")[0] == "bailey-rohru-1920"
    assert "bailey-rohru-1920" not in profile_policy.audit({})
    assert "@book{bailey1920rohru," in (DATA / "cldf/sources.bib").read_text()
    assert "roh" in {r[0] for r in csv.reader((DATA / "cldf/languages.csv").open())}
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(CSV), errors, name="20260925-bailey-rohru")
    assert not errors.getvalue()
    assert len(parsed) == stats["converted"] == 309
    raw = {r[10]: r for r in installed()}
    assert {r.entry_key for r in parsed} == set(raw)
    assert all(r.old_form == raw[r.entry_key][2] for r in parsed)


def test_historical_accepted_subset_audit_inputs_preserved():
    import hashlib
    review = json.loads((PACKAGE / "independent-sample-audit-20260926.json").read_text())
    assert review["sample_size"] == 20 and review["material_errors"] == 0
    assert Counter(int(x["page"]) for x in review["entries"]) == {127:5,128:5,129:5,130:5}
    for filename, digest in review["input_sha256"].items():
        assert hashlib.sha256((PACKAGE / filename).read_bytes()).hexdigest() == digest


def test_fresh_literal_audit_matches_installed_input():
    import hashlib
    review = json.loads((PACKAGE / "independent-literal-audit-20260926-pass2.json").read_text())
    assert len(review["sample_cells"]) == 20 and review["material_errors"] == 0
    assert Counter(r["page"] for r in review["sample_cells"]) == {"127":5,"128":5,"129":5,"130":5}
    assert hashlib.sha256((PACKAGE / "full-transcription.tsv").read_bytes()).hexdigest() == review["hashes"]["full-transcription-staged.tsv"]
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == review["hashes"]["literal-staged.csv"]
