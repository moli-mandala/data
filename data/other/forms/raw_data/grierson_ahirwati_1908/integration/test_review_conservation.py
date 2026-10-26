"""Scope regressions for the historical review scaffold, not installation acceptance."""
import csv,json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
REVISION=HERE/'review-assembly-r1'
def records(path):return [json.loads(line) for line in path.read_text().splitlines()]

def test_full_physical_scope_has_closed_emission_mapping_without_extra_hissar_heads():
    rows=list(csv.reader((REVISION/'combined-review-candidate.csv').open()))
    audit=records(REVISION/'combined-review-audit.jsonl')
    keys={r[10] for r in rows}
    assert len(rows)==len(keys)==653
    assert len(audit)==len({r['source_unit_key'] for r in audit})==1961
    assert {k for a in audit for k in a['entry_keys']}==keys
    physical=Counter(a['kind'] for a in audit if '-physical-' in a['kind'])
    assert physical=={'No37-physical-Roman':367,'No37-physical-Native':365,
                     'No38-physical-Roman':133,'No38-physical-Native':194,
                     'outside-physical-components':2}
    for a in audit:
        if a['kind'].endswith('physical-Roman'):assert a['entry_keys']
        if a['kind']=='No38-physical-Native':assert a['entry_keys']==[]
        if a['kind']=='outside-physical-components':assert a['entry_keys']==[]
    assert sum(a['kind']=='structural-parent' for a in audit)==173
    assert sum(a['kind']=='outside-eligible' for a in audit)==15

def test_documentary_and_unattested_grammar_never_become_lexical_heads():
    raw=records(PACKAGE/'grammar/child-witnesses-reviewed.jsonl')
    proposed=records(HERE/'grammar-metadata-proposal-r2/all98-keyed-grammar-metadata-proposal.jsonl')
    by={r['source_unit_key']:r for r in proposed}
    audit=records(REVISION/'combined-review-audit.jsonl')
    emitted={a['source_unit_key'] for a in audit if a['kind']=='grammar'}
    assert len(raw)==len(by)==98 and len(emitted)==75
    excluded=[]
    for r in raw:
        d=by[r['witness_key']]
        assert (r['source_form'],r['source_gloss'],r['language_label'],r['disposition'])==(d['exact_form_guard'],d['exact_gloss_guard'],d['exact_language_label_guard'],d['exact_disposition_guard'])
        if d['proposed_language_id'] is None:
            excluded.append(r)
            assert r['witness_key'] not in emitted and d['proposed_tags']==[]
    assert len(excluded)==23
    graph=records(HERE/'grammar-metadata-proposal-r2/source-graph-proposal.jsonl')
    assert Counter(g['relation'] for g in graph)=={'variant':12,'component':1}
    assert all(g['child_key'] in emitted and set(g['parent_keys'])<=emitted for g in graph)
    assert all(g['accepted'] is False for g in graph)

def test_review_graphs_and_narrow_native_remain_pending_and_regional_routes_are_not_guessed():
    rows=list(csv.reader((REVISION/'combined-review-candidate.csv').open()))
    audit=records(REVISION/'combined-review-audit.jsonl')
    assert all(not any(r[i] for i in (8,11,12,13)) for r in rows)
    assert all(a['accepted_for_installation'] is False for a in audit)
    no38=[a for a in audit if a['kind'] in ('No38-atomic','No38-expression')]
    assert len(no38)==155 and all(a['actual_row'][4]==a['actual_row'][5]=='' for a in no38)
    regional=[r for r in rows if 'Bawal-Kot-Kasam' in r[10]]
    assert len(regional)==4
    assert all('Gurgaon' not in r[14] and 'gurgaon' not in r[14] for r in regional)

def test_no38_approved_voice_scope_and_stable_expression_keys_preserve_source():
    raw=records(PACKAGE/'specimen38/native-review/atomic-emissions-proposal.jsonl')
    revised=records(HERE/'specimen38-reviewed-root-r3/atomic-emissions-reviewed.jsonl')
    assert len(raw)==len(revised)==132
    assert sum(a['tags']!=b['tags'] for a,b in zip(raw,revised))==115
    for a,b in zip(raw,revised):
        assert all(a[k]==b[k] for k in a if k!='tags')
    kahaa=[r for r in revised if r['emission_form']=='kahā' and r['physical_line']==10 and r['physical_word']==3]
    assert len(kahaa)==1 and 'pass' not in kahaa[0]['tags']
    expressions=records(HERE/'specimen38-reviewed-root-r3/expression-emissions-reviewed.jsonl')
    assert len(expressions)==23
    assert all(r['key']==r['first_word_key']+':expression' and r['historical_ordinal_key']!=r['key'] for r in expressions)
