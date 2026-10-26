"""Whole original chapter proposal; no installation or compiled build."""
import csv
import importlib.util
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / 'data/other/forms/raw_data/nair_mullu_kurumba_2013'


def rows():
    return list(csv.reader((PACKAGE / 'proposed.csv').open(newline='')))


def test_complete_scope_closed_keys_and_repeated_occurrences():
    source = [json.loads(s) for s in (PACKAGE / 'physical-units-first-reading.jsonl').read_text().splitlines()]
    audit = [json.loads(s) for s in (PACKAGE / 'proposal-audit.jsonl').read_text().splitlines()]
    actual = rows()
    assert len(source) == len(audit) == 65
    assert [u['source_unit_key'] for u in source] == [a['source_unit_key'] for a in audit]
    assert sum(len(u['source_witnesses']) for u in source) == 164
    assert len(actual) == len({r[10] for r in actual}) == 135
    assert sum(r[0] == 'Mullu Kurumba' for r in actual) == 110
    assert sum(r[0] == 'Malayalam' for r in actual) == 25
    assert {k for a in audit for k in a['entry_keys']} == {r[10] for r in actual}
    dispositions = [d for a in audit for d in a['witness_dispositions']]
    assert sum(d['disposition'] == 'exact_repeated_attestation_reuse' for d in dispositions) == 6
    assert sum(not d['entry_keys'] for d in dispositions) == 23
    assert sum(u['printed_page'] in (3, 12) for u in source) == 3


def test_original_bound_typography_is_not_mistagged_as_phrase():
    old = list(csv.reader((PACKAGE / 'pre-typography-revision/proposed.csv').open(newline='')))
    new = list(csv.reader((PACKAGE / 'revision1/proposed.csv').open(newline='')))
    assert [r[10] for r in old] == [r[10] for r in new]
    changed = []
    for a, b in zip(old, new):
        assert a[:14] == b[:14]
        if a[14] != b[14]:
            assert set(a[14].split()) - set(b[14].split()) == {'multiword-expression'}
            changed.append(b[2])
    assert set(changed) == {'- il', 'avanu –e', 'en –atti'}
    assert [r[2] for r in new if 'multiword-expression' in r[14].split()] == ['marattil ninnu']


def test_literal_notation_source_conflicts_and_separate_paradigms():
    keyed = {r[10]: r for r in rows()}
    def get(tail): return keyed['nair2013mullukurumba:' + tail]
    assert get('p55:masculine-examples:w1')[2] == 'al̲ak-en'
    assert get('p56:pronouns-first-plural:w2')[2] == 'n̲aŋŋaḷ'
    assert get('p57:relative-participle-examples:w2')[2] == 'paañña'
    conflict = get('p56:ablative-examples:w2')
    assert conflict[2:4] == ['en-attinu', 'from the trap']
    assert 'uncertain' in conflict[14].split() and '1sg' not in conflict[14].split()
    for i in range(1, 5):
        pron = get(f'p56:pronouns-third-plural:w{i}')
        assert 'pl' in pron[14].split() and not {'m', 'f'} & set(pron[14].split())
    for i in range(1, 6):
        assert {'m', 'pl', 'suffix'} <= set(get(f'p57:suffix-third-plural-cell:w{i}')[14].split())
    assert all('suffix-third-feminine-plural-blank' not in k and 'zero-cells' not in k for k in keyed)
    assert not any(':numeral' in k for k in keyed)
    assert sum(bool(r[11]) for r in keyed.values()) == 11
    for r in keyed.values():
        if r[11]: assert r[11] in keyed and r[0] == keyed[r[11]][0]
        assert not r[4] and not r[5] and not r[12] and not r[13]


def test_reviewed_revision_has_only_source_supported_metadata_and_slot_relations():
    old = list(csv.reader((PACKAGE / 'revision1/proposed.csv').open(newline='')))
    new = rows()
    assert [r[10] for r in old] == [r[10] for r in new]
    assert [r[2] for r in old] == [r[2] for r in new]
    for row in new:
        if ':present-suffixes:' in row[10] or ':future-suffixes:' in row[10]:
            assert 'verb' not in row[14].split()
            assert {'affix', 'suffix'} <= set(row[14].split())
    keyed = {r[10]: r for r in new}
    assert 'm' in keyed['nair2013mullukurumba:p55:plural-examples:w1'][14].split()
    assert keyed['nair2013mullukurumba:p56:instrumental-examples:w1'][3] == 'by the knife'
    for section, count in [('suffix-third-masculine-singular-cell',3), ('suffix-third-plural-cell',5), ('suffix-third-feminine-singular-cell',4)]:
        head = f'nair2013mullukurumba:p57:{section}:w1'
        assert not keyed[head][11]
        for index in range(2,count+1):
            child = keyed[f'nair2013mullukurumba:p57:{section}:w{index}']
            assert child[11] == head and 'alternate' in child[14].split()
            assert 'not a claim of free interchangeability' in child[6]
    continuity = json.loads((PACKAGE/'review-repair-continuity.json').read_text())
    assert continuity['changed_rows'] == 17 and continuity['changed_fields'] == 35


def test_actual_parser_retains_all_fields_and_identity_keys():
    path = PACKAGE / 'validate_actual_parser.py'
    spec = importlib.util.spec_from_file_location('mullu_source_parser', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    report = json.loads((PACKAGE / 'actual-parser-review.json').read_text())
    for normalization in ('NFC', 'NFD'):
        assert report['checks'][normalization]['rows'] == 135
        assert report['checks'][normalization]['all_15_fields_exact']
        assert report['checks'][normalization]['initial_dedupe_keys'] == 135
