"""Regression checks for the image-reviewed ambiguous index references."""
import copy
from collections import Counter
from unittest.mock import patch

import pytest

from test_yoshioka_cleanup import cleanup


@pytest.fixture(scope='module')
def reviewed():
    records = cleanup.entry_records(cleanup.load_snapshot())
    with patch.object(cleanup, 'apply_reviewed_crossreferences', lambda rows, audits: None):
        before_rows, before_audits = cleanup.compile_records(records)
    rows, audits = cleanup.compile_records(records)
    return before_rows, before_audits, rows, audits


def test_review_covers_every_previously_unresolved_form_and_preserves_other_rows(reviewed):
    old, old_audits, rows, audits = reviewed
    decisions = cleanup.crossreference_decisions()
    keys = {d['entry_key'] for d in decisions}
    unresolved = [a for a in old_audits if 'unresolved-crossreference' in a['Review']]
    assert len(unresolved) == 57
    assert len(keys) == len(decisions) == 62
    assert keys == {k for a in unresolved for k in a['Emitted_Keys'].split('|')}
    assert Counter(d['decision'] for d in decisions) == {'resolved': 56, 'ambiguous': 6}
    assert Counter(a['Crossreference_Resolution'] for a in audits if a['Crossreference']) == {
        'resolved': 214, 'partial': 2, 'unresolved': 4}
    before = {r[10]: r for r in old}
    assert len(rows) == len(before) == 4886
    for row in rows:
        original = before[row[10]]
        if row[10] not in keys:
            assert row == original
        else:
            assert all(row[i] == original[i] for i in range(15) if i not in {3, 6, 7, 11, 14})
            assert original[6] in row[6]
            assert set(original[7].split(';')) <= set(row[7].split(';'))
            assert set(original[14].split()) - {'uncertain', 'alternate'} <= set(row[14].split())
    previously_resolved = {a['Entry_Key']: a['Crossreference_Target'] for a in old_audits if a['Crossreference_Target']}
    assert len(previously_resolved) == 163
    assert all(a['Crossreference_Target'] == previously_resolved[a['Entry_Key']]
               for a in audits if a['Entry_Key'] in previously_resolved)


def test_index_lists_do_not_collapse_different_senses_or_link_ambiguous_children(reviewed):
    by = {r[10]: r for r in reviewed[2]}
    assert [by[k][3] for k in ['yoshioka-entry-749', 'yoshioka-entry-749:variant:1',
                              'yoshioka-entry-749:variant:2']] == ['chop, cut down, part', 'make bloom', 'make chop']
    assert by['yoshioka-entry-682'][11] == ''
    assert by['yoshioka-entry-682:variant:1'][11] == 'yoshioka-entry-1146'
    assert by['yoshioka-entry-726'][11] == 'yoshioka-entry-1905'
    assert by['yoshioka-entry-726:variant:1'][11] == ''
    assert by['yoshioka-entry-726:variant:1'][3] == ''
    ambiguous = [r for r in reviewed[2] if not r[3]]
    assert len(ambiguous) == 6
    assert all(not r[11] and 'uncertain' in r[14].split() and 'alternate' not in r[14].split()
               and 'Reference is ambiguous between ' in r[6] for r in ambiguous)


def test_printed_root_scope_disambiguates_identical_stems(reviewed):
    by = {r[10]: r for r in reviewed[2]}
    assert by['yoshioka-entry-737:variant:1'][2] == by['yoshioka-entry-738'][2]
    assert by['yoshioka-entry-737:variant:1'][3] == 'make open'
    assert by['yoshioka-entry-737:variant:1'][11] == 'yoshioka-entry-1062'
    assert by['yoshioka-entry-738'][3] == 'make catch, make pack'
    assert by['yoshioka-entry-738'][11] == 'yoshioka-entry-1084'
    assert by['yoshioka-entry-2566'][11] == 'yoshioka-entry-1138:variant:1'
    assert by['yoshioka-entry-2566'][3] == 'kill, make die, perform'
    assert by['yoshioka-entry-2891'][3] == 'horn'


def test_referenced_grammar_keeps_plural_and_class_scope(reviewed):
    by = {r[10]: r for r in reviewed[2]}
    adjective = by['yoshioka-entry-105']
    assert adjective[3] == 'raw, unripe'
    assert 'Referenced entry grammar: ADJ X PL -išo' in adjective[6]
    assert {'adj', 'Burushaski-class-X'} <= set(adjective[14].split())
    assert 'pl' not in adjective[14].split()
    plural = by['yoshioka-entry-696']
    assert {'verb', 'intr', 'pl'} <= set(plural[14].split())
    assert 'Referenced entry grammar: INTR PL' in plural[6]
    horn = by['yoshioka-entry-2891']
    assert {'noun', 'Burushaski-class-Y'} <= set(horn[14].split())
    assert 'pl' not in horn[14].split()
    assert 'PL -iáŋ' in horn[6]
    assert 'yoshioka2012[p. CCLXXXI, s.v. tur]' in horn[7]
    assert 'yoshioka2012[p. CCXLIII, s.v. ltur]' in horn[7]


def test_only_reviewed_terminal_hyphen_omission_is_accepted(reviewed):
    by = {r[10]: r for r in reviewed[2]}
    assert by['yoshioka-entry-831'][2] == 'duqhúlan'
    assert by['yoshioka-entry-831'][11] == 'yoshioka-entry-2379'
    assert by['yoshioka-entry-2379'][2] == 'duqhúlan-'
    assert [d['entry_key'] for d in cleanup.crossreference_decisions()
            if d['allow_missing_terminal_hyphen']] == ['yoshioka-entry-831']


@pytest.mark.parametrize('change,match', [
    ('source', 'Changed cross-reference source row'),
    ('target', 'Changed cross-reference target row'),
    ('root', 'Changed cross-reference target evidence'),
    ('inventory', 'Changed cross-reference review inventory'),
    ('selection', 'Unreviewed cross-reference target'),
])
def test_review_rejects_stale_or_unreviewed_evidence_before_any_mutation(reviewed, change, match):
    rows, audits = copy.deepcopy(reviewed[:2])
    decisions = cleanup.crossreference_decisions()
    if change in {'source', 'target'}:
        key = decisions[0]['entry_key' if change == 'source' else 'target_key']
        next(r for r in rows if r[10] == key)[3] += ' changed'
    elif change == 'root':
        key = decisions[0]['root_keys'][0]
        next(a for a in audits if a['Entry_Key'] == key)['Raw_Text'] += ' changed'
    elif change == 'inventory':
        decisions.pop()
    else:
        decisions[0]['target_key'] = 'yoshioka-entry-3'
    before = copy.deepcopy((rows, audits))
    with pytest.raises(ValueError, match=match):
        cleanup.apply_reviewed_crossreferences(rows, audits, decisions)
    assert (rows, audits) == before
