"""Source-backed regression gates for the whole-chapter review scaffold."""
import csv,hashlib,json
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def test_sharp_literal_overlay_matches_exact_independent_old_and_new_values():
    old=rows(PACKAGE/'grammar-review/reviewed-child-witnesses.jsonl')
    new=rows(PACKAGE/'grammar-review/sharp-reviewed-revision2/reviewed-child-witnesses.jsonl')
    proposal=rows(PACKAGE/'grammar-native-review/full224-sharp-review-proposal.jsonl')
    assert len(old)==len(new)==len(proposal)==224
    for a,b,p in zip(old,new,proposal):
        assert a['source_unit_key']==b['source_unit_key']==p['source_unit_key']
        assert a['form']==p['old_form'] and b['form']==p['proposed_form']
        assert a['printed_label']==p['printed_label_before'] and b['printed_label']==p['printed_label_proposed']
        for f in ['gloss','language_label','physical_parent_key','proposed_disposition','source_record']:assert a[f]==b[f]
    assert sum(a['form']!=b['form'] for a,b in zip(old,new))==35

def test_current_uncertainty_does_not_reassert_resolved_readings():
    rr=rows(PACKAGE/'grammar-review/sharp-reviewed-revision2/reviewed-child-witnesses.jsonl')
    assert sum(bool(r['typed_uncertainties']) for r in rr)==7
    assert all(u['type']!='transcription' for r in rr for u in r['typed_uncertainties'])
    label=next(r for r in rr if ':p566:present-copula:3:' in r['source_unit_key'])
    assert '3. m.' in label['printed_label'] and not label['typed_uncertainties']
    assert '2. m.' in label['source_record']['printed_label']
    assert all(r['current_native_witness']['sha256'] for r in rr)

def test_uncertain_tags_are_current_literal_or_scoped_source_analysis_union():
    rr={r['source_unit_key']:r for r in rows(PACKAGE/'grammar-review/sharp-reviewed-revision2/reviewed-child-witnesses.jsonl')}
    ts=rows(PACKAGE/'grammar-review/sharp-reviewed-revision2/grammar-tag-proposal.jsonl')
    assert len(ts)==224
    for t in ts:
        assert ('uncertain' in t['proposed_tags'])==bool(rr[t['witness_key']]['typed_uncertainties'] or t['typed_metadata_uncertainties'])

def test_join_tail_is_accounted_but_never_separate_head():
    spec=PACKAGE/'specimens/sharp-reviewed-revision'
    atoms=rows(spec/'reviewed-physical-atoms.jsonl');emissions=rows(spec/'atomic-emission-proposal.jsonl')
    assert len(atoms)==561 and len(emissions)==560
    first='grierson1906kolamibasim:p568:specimen76-line:11:word:8'
    tail='grierson1906kolamibasim:p568:specimen76-line:12:word:1'
    by={r['source_unit_key']:r for r in emissions}
    assert tail not in by and by[first]['raw_source_form']=='Bāp-phōkān'
    assert by[first]['physical_source_keys']==[first,tail]
    contributing=[k for r in emissions for k in r['physical_source_keys']]
    assert len(contributing)==len(set(contributing))==561
    assert set(contributing)=={r['source_unit_key'] for r in atoms}

def test_punctuation_partition_preserves_true_source_sequence_and_anchored_ids():
    spec=PACKAGE/'specimens/sharp-reviewed-revision'
    expressions=rows(spec/'expression-emission-proposal.jsonl')
    physical=rows(spec/'reviewed-physical-atoms.jsonl')
    assert [k for r in expressions for k in r['physical_source_keys']]==[r['source_unit_key'] for r in physical]
    assert all(r['source_unit_key']==r['first_source_key']+':expression' for r in expressions)

def test_source_flags_preserve_comma_units_in_actual_parser():
    settings=yaml.safe_load((HERE/'combined-review-candidate.yaml').read_text())
    main=settings['sources']['grierson1906kolamibasim']
    assert main['forms']['split_alternates'] is False and main['gloss']['grammar_tags'] is False
    report=json.loads((HERE/'actual-parser-review.json').read_text())
    assert report['candidate_sha256']==digest(HERE/'combined-review-candidate.csv')
    for mode in ['NFC','NFD']:
        assert report['checks'][mode]['rows']==827 and report['checks'][mode]['all15fields_exact']
        assert report['checks'][mode]['unintended_splits']==0 and report['checks'][mode]['initial_dedupe_keys']==827

def test_combined_own_keys_and_graphs_are_not_lost_or_invented():
    with (HERE/'combined-review-candidate.csv').open() as f:rr=list(csv.reader(f))
    audit=rows(HERE/'combined-review-audit.jsonl')
    keys={r[10] for r in rr}
    assert len(rr)==len(keys)==827
    assert keys<={k for a in audit for k in a['entry_keys']}
    assert sum(bool(r[11]) for r in rr)==15 and sum(bool(r[13]) for r in rr)==2
    assert all(k in keys for r in rr for k in [r[11],*r[13].split('|')] if k)
    assert all(r[4]==r[5]=='' for r in rr)
    assert all(not r[6] for r in rr if ':specimen' in r[10])
