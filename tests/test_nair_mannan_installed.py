"""Installed whole Mannan chapter checks; no compiled build."""
import csv
import importlib.util
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
PACKAGE = DATA / 'data/other/forms/raw_data/nair_mannan_2013'


def rows():
    return list(csv.reader((DATA / 'data/other/forms/20260927-nair-mannan.csv').open(newline='')))


def test_complete_installed_audit_preserves_raw_history():
    actual = rows()
    assert actual == list(csv.reader((PACKAGE / 'proposed.csv').open(newline='')))
    assert len(actual) == len({r[10] for r in actual}) == 154
    audit = [json.loads(s) for s in (PACKAGE / 'installed-audit.jsonl').read_text().splitlines()]
    raw = [json.loads(s) for s in (PACKAGE / 'proposal-audit.jsonl').read_text().splitlines()]
    assert len(audit) == 140 and [a['source_record'] for a in audit] == raw
    keys = [k for a in audit for k in a['entry_keys']]
    assert len(keys) == len(set(keys)) == 154
    assert set(keys) == {r[10] for r in actual}
    assert all(a['accepted_for_installation'] for a in audit)
    source = [json.loads(s) for s in (PACKAGE / 'physical-units-first-reading.jsonl').read_text().splitlines()]
    assert sum(len(u['source_witnesses']) for u in source) == 208
    assert sum(r[0] == 'Mannan' for r in actual) == 123
    assert sum(r[0] == 'Malayalam' for r in actual) == 31


def test_source_metadata_repair_is_tags_only_and_preserves_literals():
    old = list(csv.reader((PACKAGE / 'revision1/proposed.csv').open(newline='')))
    current = rows()
    assert len(old) == len(current)
    assert all(a[:14] == b[:14] for a, b in zip(old, current))
    assert sum(a[14] != b[14] for a, b in zip(old, current)) == 22
    for row in current:
        if ':nonpast-negative:' in row[10]:
            assert 'non-past' in row[14].split()
        if ':kinship:' in row[10] or ':kin-plural:' in row[10]:
            assert 'kinship' in row[14].split()
    keyed = {r[10]: r for r in current}
    assert keyed['nair2013mannan:p48:vocabulary:23:w1'][14].split() == ['verb']
    assert keyed['nair2013mannan:p47:past-negative:0:w1'][2] == 'aa'
    assert keyed['nair2013mannan:p46:nasal-clusters:1:w1'][2] == 'pan̲n̲i'
    assert all(not r[11] and not r[12] and not r[13] for r in current)


def test_actual_canonical_parser_metadata_identity_and_repeat_noop(monkeypatch):
    folder = PACKAGE / 'installation'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('mannan_canonical_checks', folder / 'check_canonical.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.main()
    assert report['parser_monkeypatch'] is False
    assert report['checks']['NFC']['initial_dedupe_keys'] == 154
    assert report['checks']['NFD']['all15fields_exact']
