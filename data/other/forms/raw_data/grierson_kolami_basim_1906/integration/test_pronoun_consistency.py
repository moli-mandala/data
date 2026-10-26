"""Exact cross-section source-policy repair without lexical or sample drift."""
import csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
def rr(p):
    with p.open() as f:return list(csv.reader(f))
def test_nine_exact_approved_obj_only_removals_and_all_other_fields():
    old=rr(HERE/'revision4/combined-review-candidate.csv');new=rr(HERE/'revision5/combined-review-candidate.csv')
    a=json.loads((PACKAGE/'root-independent-review/grammar-object-scope-root-adjudication.json').read_text())
    decisions={d['source_unit_key']:d for d in a['decisions']};assert len(decisions)==9
    changed=set()
    for x,y in zip(old,new):
        assert x[:14]==y[:14]
        if x!=y:
            changed.add(x[10]);d=decisions[x[10]]
            assert x[14].split()==d['old_tags'] and y[14].split()==d['new_tags']
            assert y[14].split()==[t for t in x[14].split() if t!='obj']
    assert len(old)==len(new)==827 and changed==set(decisions)
    assert all('obj' not in r[14].split() for r in new)
def test_same20_sample_source_keys_and_literals_are_retained():
    old=json.loads((HERE/'revision4/fresh-output-selection-20260927.json').read_text())
    new=json.loads((HERE/'revision5/fresh-output-selection-20260927.json').read_text())
    assert old['seed']==new['seed']==20260927
    for field in ['sample','targeted_controls']:
        assert [r['source_unit_key'] for r in old[field]]==[r['source_unit_key'] for r in new[field]]
        assert all(x['actual_row'][:14]==y['actual_row'][:14] for x,y in zip(old[field],new[field]))
    assert len(new['sample'])==20 and len(new['literal_repair_regression_controls'])==59
def test_current_tag_and_graph_guards_match_exact_repair():
    before=PACKAGE/'grammar-review/sharp-reviewed-revision3'
    after=PACKAGE/'grammar-review/whole-metadata-consistency-revision'
    for filename,field in [('grammar-tag-proposal.jsonl','proposed_tags'),('grammar-graph-proposal.jsonl','proposed_tags_guard')]:
        old=[json.loads(s) for s in (before/filename).read_text().splitlines()];new=[json.loads(s) for s in (after/filename).read_text().splitlines()]
        assert [r['witness_key'] for r in old]==[r['witness_key'] for r in new]
        changed=0
        for x,y in zip(old,new):
            if 'obj' in x[field]:
                assert y[field]==[t for t in x[field] if t!='obj'];changed+=1
            else:assert y==x
            if filename.startswith('grammar-graph'):
                for f in ['variant_of_witness_key','ordered_parents','source_claim','scope_caveat']:assert y[f]==x[f]
        assert changed==9
def test_final_parser_pins_current_candidate():
    report=json.loads((HERE/'revision5/actual-parser-review.json').read_text())
    assert report['candidate_sha256']==hashlib.sha256((HERE/'revision5/combined-review-candidate.csv').read_bytes()).hexdigest()
    for mode in ['NFC','NFD']:assert report['checks'][mode]['all15fields_exact'] and report['checks'][mode]['rows']==827
