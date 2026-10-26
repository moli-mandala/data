"""Focused proposal checks for the complete three-page chapter."""
import csv
import json
import unicodedata
from pathlib import Path

import yaml
from pybtex.database import parse_file
from source_meta import SourceMeta
from tags import GRAMMATICAL_TAGS, GENDER_TAGS

P = Path(__file__).resolve().parent


def corpus():
    units = [json.loads(s) for s in (P / "physical-units-first-reading.jsonl").read_text().splitlines()]
    audit = [json.loads(s) for s in (P / "proposal-audit.jsonl").read_text().splitlines()]
    rows = list(csv.reader((P / "proposed.csv").open()))
    return units, audit, rows


def test_whole_chapter_physical_census_and_controls():
    units, audit, rows = corpus()
    assert {u["printed_page"] for u in units} == {37, 38, 39}
    assert len(units) == len(audit) == 79 and len(rows) == 67
    assert sum(a["disposition"] == "audit-only" for a in audit) == 12
    assert sum(u["scope"] == "target" and bool(u["malayalam_control"]) for u in units) == 53
    assert {a["unit_key"] for a in audit} == {u["unit_key"] for u in units}
    assert len({r[10] for r in rows}) == len(rows)
    assert all(a["disposition"] == "audit-only" for a in audit if a["scope"] == "malayalam-control")
    assert all(a["disposition"] == "audit-only" for a in audit if a["scope"] != "target")
    by = {r[10]: r for r in rows}
    assert all("Printed Malayalam comparison: " + u["malayalam_control"] in by[u["unit_key"]][6]
               for u in units if u["scope"] == "target" and u["malayalam_control"])


def test_source_contradictions_and_distinct_homographs_retained():
    _, audit, rows = corpus()
    by = {r[10]: r for r in rows}
    p = "nair2013malamuthan:p37:"
    assert by[p + "final-n:01"][2:4] == ["cekkaŋ", "boy"]
    assert by[p + "final-n-formal:01"][2:4] == ["cekkam", "boy"]
    assert "formal n/m conflict" in by[p + "final-n-formal:01"][6]
    q = "nair2013malamuthan:p38:pronouns:"
    same = [r for r in rows if r[2] == "ibaḷu" and r[10].startswith(q)]
    assert len(same) == 2 and {r[3] for r in same} == {"she", "they"}
    assert any(a["section"] == "past-uttu" and a["scope"] == "comparison-context" for a in audit)
    assert by["nair2013malamuthan:p38:demonstratives:03"][2] == "-nee"


def test_literal_profile_tags_and_graph_policy():
    _, _, rows = corpus()
    assert all(len(r) == 15 and r[0] == "Malamuthan" and r[2] for r in rows)
    assert all(unicodedata.is_normalized("NFC", c) and "\ufffd" not in c for r in rows for c in r)
    assert all(not r[4] and not r[5] and not r[8] and not r[9] and not r[12] and not r[13] for r in rows)
    graph = json.loads((P / "graph-decisions-proposal.json").read_text())
    variants = {e["child"]: e["parent"] for e in graph["variant_edges"]}
    assert len(variants) == 2
    assert {r[10]: r[11] for r in rows if r[11]} == variants
    assert all(set(r[14].split()) <= (GRAMMATICAL_TAGS | GENDER_TAGS) for r in rows)
    assert all(len(r[14].split()) == len(set(r[14].split())) for r in rows)
    assert all(r[7].startswith("nair2013malamuthan[p. ") for r in rows)


def test_proposed_metadata_and_reference_parse():
    settings = P / "20260927-nair-malamuthan.yaml"
    y = yaml.safe_load(settings.read_text())
    assert y["defaults"]["transcription"] == {"convert": False, "preserve_hyphens": True}
    assert y["defaults"]["identity"]["dedupe_by_entry_key"] is True
    assert y["sources"]["nair2013malamuthan"]["forms"]["split_alternates"] is False
    assert "nair2013malamuthan" in SourceMeta([settings]).sources
    assert set(parse_file(str(P / "proposed-reference.bib")).entries) == {
        "nair2013malamuthan", "somasekharannair1981malamuthan"}
    lang = list(csv.DictReader((P / "proposed-language.csv").open()))
    assert len(lang) == 1 and lang[0]["ID"] == "Malamuthan"
    assert not lang[0]["Glottocode"] and not lang[0]["Latitude"] and not lang[0]["Longitude"]


def test_scoped_generic_parser_preserves_exact_form_and_key():
    import make_cldf
    rows = list(csv.reader((P / "proposed.csv").open()))
    errors = []
    parsed, stats = make_cldf.parse_file(str(P / "proposed.csv"), errors)
    assert len(parsed) == 67 and not errors and stats["converted"] == 0
    raw = {r[10]: r for r in rows}
    assert {r.entry_key for r in parsed} == set(raw)
    assert all(r.form == raw[r.entry_key][2] and r.old_form == raw[r.entry_key][2]
               and not r.ipa and r.variant_of_key == raw[r.entry_key][11] for r in parsed)


def test_installed_source_stage_exact_parity_when_receipt_exists():
    import hashlib
    import pytest
    import make_cldf
    receipt_path = P / "installation-receipt.json"
    if not receipt_path.exists():
        pytest.skip("Source-stage installation has not been independently approved and run")
    data_root = next(p for p in P.parents if (p / "tags.py").exists())
    canonical = data_root / "data/other/forms/20260927-nair-malamuthan.csv"
    config = data_root / "data/other/forms/20260927-nair-malamuthan.yaml"
    receipt = json.loads(receipt_path.read_text())
    assert canonical.read_bytes() == (P / "proposed.csv").read_bytes()
    assert config.read_bytes() == (P / "20260927-nair-malamuthan.yaml").read_bytes()
    assert receipt["rows"] == 67 and receipt["audit_units"] == 79
    assert len((P / "installed-audit.jsonl").read_text().splitlines()) == 79
    assert hashlib.sha256((data_root / "data/form-identities.csv").read_bytes()).hexdigest() == receipt["durable_identity_sha256"]
    assert any(r["ID"] == "Malamuthan" and not r["Glottocode"] and not r["Latitude"]
               for r in csv.DictReader((data_root / "cldf/languages.csv").open()))
    assert {"nair2013malamuthan", "somasekharannair1981malamuthan"} <= set(
        parse_file(str(data_root / "cldf/sources.bib")).entries)
    errors = []
    parsed, stats = make_cldf.parse_file(str(canonical), errors)
    assert len(parsed) == 67 and not errors and stats["converted"] == 0
