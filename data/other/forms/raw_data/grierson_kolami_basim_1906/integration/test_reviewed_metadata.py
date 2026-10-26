"""Independent-source recommendations must survive the explicit revision chain."""
import csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def csvrows(p):
    with p.open() as f:return list(csv.reader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def test_exact_approved_tag_scope_and_all_protected_fields():
    old=csvrows(HERE/'combined-review-candidate.csv');new=csvrows(HERE/'revision2/combined-review-candidate.csv')
    recs=rows(PACKAGE/'specimen-metadata-independent-review/keyed-tag-recommendations.jsonl')
    approved=json.loads((PACKAGE/'root-independent-review/specimen-metadata-root-adjudication.json').read_text())
    assert len(old)==len(new)==827 and len(recs)==138
    rb={r['source_unit_key']:r for r in recs}
    assert set(rb)==set(approved['accepted_source_unit_keys'])
    changed=set()
    for a,b in zip(old,new):
        assert a[:14]==b[:14]
        if a[14]!=b[14]:
            changed.add(a[10]);r=rb[a[10]]
            assert a[2]==r['exact_form_guard'] and a[3]==r['exact_raw_gloss_guard']
            assert set(b[14].split())==set(r['new'])|{t for t in a[14].split() if t=='uncertain' or t.startswith('dialect:')}
    assert changed==set(rb)
def test_infinitive_and_directive_classes_follow_original_scope():
    rr={r[10]:r for r in csvrows(HERE/'revision3/combined-review-candidate.csv')}
    recs=rows(PACKAGE/'specimen-metadata-independent-review/keyed-tag-recommendations.jsonl')
    infinitives=[r for r in recs if r['exact_raw_gloss_guard'].lower().strip('.,;') in {'to-feed','to-say','to-entreat','to-spend','to-eat'}]
    assert {r['exact_raw_gloss_guard'].lower().strip('.,;') for r in infinitives}=={'to-feed','to-say','to-entreat','to-spend','to-eat'}
    for r in infinitives:assert {'verb','inf'}<=set(rr[r['source_unit_key']][14].split())
    directives=[r for r in recs if 'impv' in r['new'] and 'put' in r['exact_raw_gloss_guard'].lower()]
    assert directives
    for r in directives:
        tags=set(rr[r['source_unit_key']][14].split());assert 'impv' in tags and not {'loc','pret'}&tags
    dancing=[r for r in recs if 'dancing' in r['exact_raw_gloss_guard'].lower()]
    assert dancing
    for r in dancing:assert 'verb' not in rr[r['source_unit_key']][14].split()
def test_three_current_commentaries_keep_exact_historical_text_and_no_other_delta():
    old=csvrows(HERE/'revision2/combined-review-candidate.csv');new=csvrows(HERE/'revision3/combined-review-candidate.csv')
    findings=json.loads((PACKAGE/'grammar-native-review/sharp-revision2-history-review.json').read_text())['keyed_display_findings']
    expected={d['key']:d for d in findings};changed=set()
    for a,b in zip(old,new):
        assert a[:6]+a[7:]==b[:6]+b[7:]
        if a[6]!=b[6]:
            changed.add(a[10]);d=expected[a[10]]
            assert a[6].count(d['old'])==1 and b[6]==a[6].replace(d['old'],d['new'])
    assert changed==set(expected)
    children={r['source_unit_key']:r for r in rows(PACKAGE/'grammar-review/sharp-reviewed-revision3/reviewed-child-witnesses.jsonl')}
    graphs={r['witness_key']:r for r in rows(PACKAGE/'grammar-review/sharp-reviewed-revision3/grammar-graph-proposal.jsonl')}
    for d in findings:
        r=children[d['key']] if d['field']=='qualification' else graphs[d['key']]
        assert r[d['field']]==d['new'] and r['historical_display_commentary'][d['field']]==d['old']
def test_revised_actual_parser_all_fields_and_relations():
    report=json.loads((HERE/'revision3/actual-parser-review.json').read_text())
    assert report['candidate_sha256']==sha(HERE/'revision3/combined-review-candidate.csv')
    assert report['monkeypatch'] is False and report['canonical_changes'] is False
    for mode in ['NFC','NFD']:
        assert report['checks'][mode]['all15fields_exact'] and report['checks'][mode]['rows']==827
        assert report['checks'][mode]['initial_dedupe_keys']==827 and report['checks'][mode]['unintended_splits']==0
    assert report['source_local_graph']=={'variant_links':15,'component_sets':2,'ordered_component_links':4,'closed':True,'acyclic':True}

def test_final_all560_source_scopes_and_exact15_scholarly_notes():
    rr={r[10]:r for r in csvrows(HERE/'revision4/combined-review-candidate.csv')}
    original={r[10]:r for r in csvrows(HERE/'combined-review-candidate.csv')}
    final=rows(PACKAGE/'specimen-metadata-independent-review/residual-revision3/all-560-final-metadata-proposal.jsonl')
    approval=json.loads((PACKAGE/'root-independent-review/residual-metadata-root-adjudication.json').read_text())
    assert len(final)==560 and len(approval['notes_decisions'])==15
    for m in final:
        r=rr[m['source_unit_key']]
        assert r[2]==m['exact_form_guard'] and r[3]==m['exact_raw_gloss_guard']
        assert set(r[14].split())==set(m['proposed_tags'])
        if m['typed_metadata_uncertainties']:assert 'uncertain' in r[14].split()
    for d in approval['notes_decisions']:
        r=rr[d['source_unit_key']]
        assert r[6]==d['new_notes'] and original[r[10]][6]==d['old_notes']
    assert sum(bool(m['typed_metadata_uncertainties']) for m in final)==14
    assert sum(original[k][14]!=r[14] for k,r in rr.items())==214
    assert sum(original[k][6]!=r[6] for k,r in rr.items())==18
    assert all(original[k][:6]+original[k][7:14]==r[:6]+r[7:14] for k,r in rr.items())

def test_contextual_case_nonfinite_and_lexicalized_adverb_boundaries():
    rr={r[10]:r for r in csvrows(HERE/'revision4/combined-review-candidate.csv')}
    prefix='grierson1906kolamibasim:'
    for suffix in ['p565:specimen75-line:4:word:1','p565:specimen75-line:18:word:3']:
        r=rr[prefix+suffix];assert r[3] in {'again','again,'} and 'adv' in r[14].split()
        assert not {'verb','participle','conjunctive-participle'}&set(r[14].split())
    r=rr[prefix+'p564:specimen75-line:10:word:4']
    assert {'verb','participle','conjunctive-participle'}<=set(r[14].split())
    r=rr[prefix+'p564:specimen75-line:16:word:3']
    assert r[2]=='mākkun' and 'dat' in r[14].split() and 'loc' not in r[14].split()
    assert 'accusative/dative' in r[6] and r[3]=='neck-on'
    r=rr[prefix+'p568:specimen76-line:11:word:8']
    assert r[2]=='Bāp-phōkān' and not {'loc','abl','dat','obj'}&set(r[14].split())
    assert 'uncertain' in r[14].split() and 'separate case analysis' in r[6]

def test_final_parser_exact_key_closure_and_profile_rare_symbols():
    report=json.loads((HERE/'revision4/actual-parser-review.json').read_text())
    assert report['candidate_sha256']==sha(HERE/'revision4/combined-review-candidate.csv')
    for normalization in ['NFC','NFD']:
        assert report['checks'][normalization]['all15fields_exact'] and report['checks'][normalization]['initial_dedupe_keys']==827
    rr=csvrows(HERE/'revision4/combined-review-candidate.csv')
    assert any('ï' in r[2] for r in rr)
    assert report['terminal_display_delta_count']==155 and report['source_local_graph']['acyclic']
