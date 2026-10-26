"""Focused checks for the full seven-page Betta Kurumba proposal."""
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


def test_whole_source_census_and_controls():
    units, audit, rows = corpus()
    assert {u["printed_page"] for u in units} == set(range(12, 19))
    assert len(units) == len(audit) == 225 and len(rows) == 206
    assert sum(u["role"] == "target" for u in units) == 203
    assert sum(a["disposition"] == "audit-only" for a in audit) == 22
    assert sum(bool(u["surface"]) for u in units) == 34
    assert sum(u["section"] == "person-suffix-table" for u in units) == 12
    assert sum(u["role"] == "zero-exponent" for u in units) == 5
    assert sum(u["role"] == "blank" for u in units) == 1
    assert {a["unit_key"] for a in audit} == {u["unit_key"] for u in units}
    assert all(a["proposed_entry_keys"] or a["exclusion_reason"] for a in audit)
    assert all(a["proposed_entry_keys"] or a["role"] != "target" for a in audit)


def test_literal_variation_and_source_anomalies_retained():
    _, audit, rows = corpus()
    by = {r[10]: r for r in rows}
    p = "nair2013bettakurumba:"
    assert by[p + "p12:diagnostic-case-markers:02"][2] == "-ooda"
    assert by[p + "p13:instrumental-marker:01"][2] == "ooḍa"
    assert by[p + "p13:plural-examples:02"][2] == "algu"
    assert "aaḷ-g" in by[p + "p13:plural-examples:02"][6]
    assert by[p + "p16:durative-negative-markers:01"][2] == "-lpad-"
    assert by[p + "p16:durative-negative-examples:01"][2] == "baar-lpaḍ-a"
    assert by[p + "p16:relative-participle-examples:02"][2] == "baar-lpad-a"
    assert by[p + "p17:verbal-participle-examples:04"][2] == "ba-nd-um"
    assert "conflicts" in by[p + "p17:verbal-participle-examples:04"][6]
    assert by[p + "p18:completive-examples:02"][2] == "ooyḍṇaacaa"
    assert any(a["role"] == "blank" and a["gloss"] == "for (the purpose of) speaking" for a in audit)
    assert any(a["role"] == "attribution-held" and a["form"] == "Bettu" for a in audit)


def test_tags_graph_and_no_inferred_ipa():
    _, _, rows = corpus()
    assert all(len(r) == 15 and r[0] == "BettaKurumba" and r[2] for r in rows)
    assert all(unicodedata.is_normalized("NFC", c) and "\ufffd" not in c for r in rows for c in r)
    assert all(not r[4] and not r[5] and not r[8] and not r[9] and not r[12] and not r[13] for r in rows)
    assert all(set(r[14].split()) <= (GRAMMATICAL_TAGS | GENDER_TAGS) for r in rows)
    assert all(len(r[14].split()) == len(set(r[14].split())) for r in rows)
    keys = {r[10] for r in rows}
    assert len(keys) == 206
    graph = json.loads((P / "graph-decisions-proposal.json").read_text())
    variants = {e["child"]: e["parent"] for e in graph["variant_edges"]}
    assert len(variants) == 3
    assert {r[10]: r[11] for r in rows if r[11]} == variants
    assert set(variants.values()) <= keys
    assert not graph["derivation_edges"] and not graph["borrowed_edges"]


def test_drawn_low_lines_and_literal_dash_occurrences():
    units, audit, rows = corpus()
    by_unit = {u["unit_key"]: u for u in units}
    by_row = {r[10]: r for r in rows}
    census = json.loads((P / "lowline-vector-census.json").read_text())
    assert len(census["marks"]) == 20
    assert census["page_counts"] == {"12": 0, "13": 1, "14": 0, "15": 3, "16": 1, "17": 8, "18": 7}
    for change in json.loads((P / "lowline-corrections.json").read_text())["changes"]:
        u = by_unit[change["unit_key"]]
        assert u[change["field"]] == change["new"]
        row = by_row[change["unit_key"]]
        if change["field"] == "surface" or not u["surface"]:
            assert row[2] == change["new"]
        else:
            assert change["new"] in row[6]
    assert by_row["nair2013bettakurumba:p12:diagnostic-case-markers:07"][2] == "–puli"
    assert by_row["nair2013bettakurumba:p15:indefinite-marker:01"][2] == "ang"
    assert by_row["nair2013bettakurumba:p17:past-definitive-examples:01"][2] == "trad̲uṭṭi"
    assert by_row["nair2013bettakurumba:p18:durative-examples:01"][2] == "bandunid̲d̲i"


def test_grammar_is_structured_and_not_lexical_gloss():
    units, audit, rows = corpus()
    by = {r[10]: r for r in rows}
    p = "nair2013bettakurumba:"
    assert by[p + "p12:diagnostic-case-markers:01"][3] == ""
    assert "acc" in by[p + "p12:diagnostic-case-markers:01"][14].split()
    assert by[p + "p13:accusative-examples:01"][3] == "tiger"
    assert by[p + "p13:dative-examples:03"][3] == "house"
    assert by[p + "p15:nonpast-examples:05"][3] == "will kill"
    assert "2sg" in by[p + "p15:nonpast-examples:05"][14].split()
    assert {"pret", "2pl"} <= set(by[p + "p16:general-negative-examples:02"][14].split())
    assert all("relative" in by[p + "p16:relative-participle-markers:" + n][14].split() for n in ("01", "02"))
    assert all(not r[3] and {"suffix", "2pl", "non-past"} <= set(r[14].split())
               for r in rows if r[10].startswith(p + "p18:person-suffix-table:08"))
    assert "neg" not in by[p + "p16:past-negative-markers:03"][14].split()
    assert "conditional" in by[p + "p16:past-negative-markers:03"][14].split()
    assert "neg" not in by[p + "p16:past-negative-markers:04"][14].split()
    a = next(a for a in audit if a["unit_key"] == p + "p13:accusative-examples:01")
    assert a["gloss"] == "tiger (Acc.)" and a["proposed_emissions"][0]["lexical_gloss"] == "tiger"


def test_source_explicit_sociative_uses_registered_comitative():
    _, _, rows = corpus()
    by = {r[10]: r for r in rows}
    p = "nair2013bettakurumba:"
    assert set(by[p + "p12:diagnostic-case-markers:03"][14].split()) == {"suffix", "comitative"}
    assert "comitative" in by[p + "p13:sociative-sentence:01"][14].split()
    assert "instr" not in by[p + "p13:sociative-sentence:01"][14].split()
    # The source labels the mixed with/in p14 forms as locative, not sociative.
    assert "comitative" not in by[p + "p14:locative-pili-examples:01"][14].split()


def test_metadata_parse_and_existing_language_route():
    y = yaml.safe_load((P / "20260927-nair-betta-kurumba.yaml").read_text())
    assert y["defaults"]["transcription"] == {"convert": False, "preserve_hyphens": True}
    assert y["defaults"]["identity"]["dedupe_by_entry_key"] is True
    assert y["sources"]["nair2013bettakurumba"]["forms"]["split_alternates"] is False
    assert "nair2013bettakurumba" in SourceMeta([P / "20260927-nair-betta-kurumba.yaml"]).sources
    assert set(parse_file(str(P / "proposed-reference.bib")).entries) == {
        "nair2013bettakurumba", "jayapal1978kurumba"}
    data_root = next(q for q in P.parents if (q / "tags.py").exists())
    languages = list(csv.DictReader((data_root / "cldf/languages.csv").open()))
    assert sum(r["ID"] == "BettaKurumba" and r["Glottocode"] == "bett1235" for r in languages) == 1


def test_scoped_generic_parser_exact_forms_and_keys():
    import make_cldf
    _, _, rows = corpus()
    errors = []
    parsed, stats = make_cldf.parse_file(str(P / "proposed.csv"), errors)
    assert len(parsed) == 206 and not errors and stats["converted"] == 0
    raw = {r[10]: r for r in rows}
    assert {r.entry_key for r in parsed} == set(raw)
    assert all(r.form == raw[r.entry_key][2] and r.old_form == raw[r.entry_key][2]
               and not r.ipa and r.variant_of_key == raw[r.entry_key][11] for r in parsed)


def test_installed_noop_allows_unrelated_shared_bibliography_append(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("betta_installer", P / "install_source.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    targets = {r: tmp_path / r for r in ("csv", "yaml", "audit", "bib")}
    for role in ("csv", "yaml", "audit"):
        targets[role].write_text(role)
    targets["bib"].write_text((P / "proposed-reference.bib").read_text() +
                             '\n@misc{unrelated_later_source, title={Later source}}\n')
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"outputs": {r: installer.current(p) for r, p in targets.items()},
                                   "durable_identity_sha256": "historical identity"}))
    monkeypatch.setattr(installer, "TARGETS", targets)
    monkeypatch.setattr(installer, "RECEIPT", receipt)
    monkeypatch.setattr(installer, "validate_approval", lambda: {"status": "approved"})
    before = {r: p.read_bytes() for r, p in targets.items()}
    installer.main(install=True)
    assert before == {r: p.read_bytes() for r, p in targets.items()}
    targets["bib"].write_text(targets["bib"].read_text().replace("Jayapal", "ChangedAuthor"))
    import pytest
    with pytest.raises(AssertionError, match="source bibliography drift"):
        installer.main(install=True)
