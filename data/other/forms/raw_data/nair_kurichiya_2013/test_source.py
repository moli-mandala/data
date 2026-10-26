"""Focused source-stage checks for the complete Kurichiya chapter."""
import csv
import io
import json
import hashlib
import unicodedata
from pathlib import Path

from pybtex.database import parse_file
import pybtex
import make_cldf
from source_meta import SourceMeta
import source_meta
from tags import GENDER_TAGS, GRAMMATICAL_TAGS

P = Path(__file__).resolve().parent


def records():
    rows = list(csv.reader((P / "proposed.csv").open()))
    audit = [json.loads(s) for s in (P / "proposal-audit.jsonl").read_text().splitlines()]
    return rows, audit


def test_complete_physical_inventory_and_dispositions():
    rows, audit = records()
    assert len(audit) == 117 and len({a["witness_key"] for a in audit}) == 117
    assert {a["pdf_page"] for a in audit} == {34, 35, 36}
    assert len(rows) == len({r[10] for r in rows}) == 118
    assert sum(r[0] == "Kurichiya" for r in rows) == 71
    assert sum(r[0] == "Malayalam" for r in rows) == 47
    assert sum(a["disposition"] == "audit-only repeated prose" for a in audit) == 1
    assert sum(len(a.get("proposed_entry_keys", [])) == 2 for a in audit) == 2
    assert {k for a in audit for k in a.get("proposed_entry_keys", [])} == {r[10] for r in rows}


def test_source_specific_grammar_and_literal_notation():
    rows, _ = records()
    by = {r[10]: r for r in rows}
    p = "nair2013kurichiya:"
    assert by[p + "p36:pronouns:I-gen-pl:1"][14].split() == ["pron", "1pl", "pl", "inclusive", "uncertain"]
    assert "genitive-row alignment" in by[p + "p36:pronouns:I-gen-pl:1"][6]
    assert by[p + "p36:pronouns:III-mas-sg:2"][2] == by[p + "p36:pronouns:III-fem-sg:1"][2]
    assert " m" not in by[p + "p36:pronouns:III-mas-sg:2"][14]
    assert by[p + "p35:question-ending"][2] == "laa"  # no invented printed hyphen
    assert by[p + "p36:utterance-1"][3] == by[p + "p36:utterance-2"][3] == "all of you should come"
    assert all(len(r) == 15 and unicodedata.is_normalized("NFC", r[2]) for r in rows)
    assert all(not r[4] and not r[5] and not r[12] and not r[13] for r in rows)
    graph = json.loads((P / "graph-review-proposal.json").read_text())
    expected = {e["child"]: e["parent"] for e in graph["proposed_variant_pairs"]}
    assert len(expected) == 8 and {r[10]: r[11] for r in rows if r[11]} == expected
    assert set(expected.values()) <= {r[10] for r in rows}
    assert all(set(r[14].split()) <= GRAMMATICAL_TAGS | GENDER_TAGS for r in rows)
    assert all("\ufffd" not in cell and not any("\ue000" <= c <= "\uf8ff" for c in cell) for r in rows for cell in r)
    assert sum(r[2].count("\u0332") for r in rows) == 12


def test_proposed_profile_bibliography_and_language_identity():
    assert "nair2013kurichiya" in SourceMeta([P / "20260927-nair-kurichiya.yaml"]).sources
    refs = parse_file(str(P / "proposed-reference.bib"))
    assert set(refs.entries) == {"nair2013kurichiya", "vayaleri1992kurichiya"}
    for entry in refs.entries.values():
        formatted = pybtex.PybtexEngine().format_from_string(entry.to_string("bibtex"), "plain", output_backend="markdown")
        assert formatted
    language = next(csv.DictReader((P / "proposed-language.csv").open()))
    assert language["ID"] == "Kurichiya" and language["Glottocode"] == "kuri1256"
    assert not language["Latitude"] and not language["Longitude"]


def test_actual_parser_preserves_literal_source_fields(monkeypatch):
    meta = SourceMeta([P / "20260927-nair-kurichiya.yaml"])
    monkeypatch.setattr(make_cldf.source_meta, "load", lambda: meta)
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(P / "proposed.csv"), errors, name="20260927-nair-kurichiya")
    raw, _ = records()
    assert not errors.getvalue() and len(parsed) == len(raw) == 118
    assert stats == {"converted": 0, "for_conversion": 0}
    assert all(p.form == r[2] and p.source == r[7] and p.entry_key == r[10] for p, r in zip(parsed, raw))


def test_installed_assets_and_global_metadata():
    # raw_data package -> forms directory
    installed = P.parent.parent / "20260927-nair-kurichiya.csv"
    profile = P.parent.parent / "20260927-nair-kurichiya.yaml"
    receipt = json.loads((P / "installation-receipt.json").read_text())
    assert hashlib.sha256(installed.read_bytes()).hexdigest() == receipt["outputs"]["csv"]
    assert installed.read_bytes() == (P / "proposed.csv").read_bytes()
    assert hashlib.sha256(profile.read_bytes()).hexdigest() == receipt["outputs"]["yaml"]
    meta = source_meta.load()
    for lang in ("Kurichiya", "Malayalam"):
        assert meta.transcription("nair2013kurichiya", installed, lang) == (None, False)
    assert meta.file_flag(installed, "identity", "dedupe_by_entry_key") is True
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(installed), errors, name="20260927-nair-kurichiya")
    assert not errors.getvalue() and len(parsed) == 118
    assert stats == {"converted": 0, "for_conversion": 0}
