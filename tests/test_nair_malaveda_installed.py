"""Source-grounded installed chapter checks; no CLDF/database build."""
import csv
import importlib.util
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/nair_malaveda_2013'
CSV = DATA / 'data/other/forms/20260927-nair-malaveda.csv'


def rows():
    return list(csv.reader(CSV.open(newline='')))


def test_complete_audit_and_historical_source_records():
    actual = rows()
    proposal = list(csv.reader((PACKAGE / 'proposed.csv').open(newline='')))
    assert actual == proposal and len(actual) == 223
    accepted = [json.loads(s) for s in (PACKAGE / 'installed-audit.jsonl').read_text().splitlines()]
    raw = [json.loads(s) for s in (PACKAGE / 'proposal-audit.jsonl').read_text().splitlines()]
    assert len(accepted) == len(raw) == 240
    assert [a['source_record'] for a in accepted] == raw
    assert len({a['source_unit_key'] for a in accepted}) == 57
    keys = [k for a in accepted for k in a['entry_keys']]
    assert len(keys) == len(set(keys)) == 223
    assert set(keys) == {r[10] for r in actual}
    assert sum(not a['entry_keys'] for a in accepted) == 17


def test_source_explicit_case_metadata_and_distinct_uncertainties():
    keyed = {r[10]: r for r in rows()}
    def row(tail): return keyed['nair2013malaveda:' + tail]
    assert {'pron', 'acc', '2pl'} <= set(row('p41:accusative-examples:w2')[14].split())
    assert {'pron', 'gen', '2sg', 'poss'} <= set(row('p41:genitive-examples:w4')[14].split())
    assert {'pron', 'gen', '3pl', 'poss'} <= set(row('p41:genitive-examples:w1')[14].split())
    assert 'kinship' in row('p41:locative-examples:w2')[14].split()
    assert 'poss' not in row('p41:locative-examples:w7')[14].split()
    assert 'uncertain' in row('p42:personal-terminations:w6')[14].split()
    assert row('p41:numerals:w9')[3] == 'fifty'
    assert 'uncertain' in row('p41:numerals:w9')[14].split()


def test_source_variants_and_literal_notation():
    actual = rows()
    keyed = {r[10]: r for r in actual}
    edges = json.loads((PACKAGE / 'graph-proposal.json').read_text())['proposals']
    assert sum(bool(r[11]) for r in actual) == len(edges) == 12
    for edge in edges:
        child, parent = keyed[edge['child_key']], keyed[edge['variant_of_key']]
        assert child[11] == parent[10]
        assert child[2] == edge['child_form'] and parent[2] == edge['head_form']
    assert sum(r[2].count('\u0332') for r in actual) == 14
    assert all(not r[1] and not r[4] and not r[5] and not r[12] and not r[13] for r in actual)


def test_actual_canonical_source_meta_parser_and_repeat_noop(monkeypatch):
    folder = PACKAGE / 'installation'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('malaveda_canonical_checks', folder / 'check_canonical.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.main()
    assert report['parser_monkeypatch'] is False
    assert report['checks']['NFC']['rows'] == report['checks']['NFD']['rows'] == 223
    assert report['checks']['NFC']['initial_dedupe_keys'] == report['checks']['NFD']['initial_dedupe_keys'] == 223
