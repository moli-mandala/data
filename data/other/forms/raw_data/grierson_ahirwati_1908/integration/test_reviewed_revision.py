"""Source-grounded whole-review regressions; no installation acceptance asserted."""
import csv,json
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
FINAL=HERE/'review-assembly-r2'
def records(p):return [json.loads(s) for s in p.read_text().splitlines()]
def rows(p):return list(csv.reader(p.open()))

def test_registered_routes_joint_regional_scope_and_source_slot_graph():
    rr=rows(FINAL/'combined-review-candidate.csv');by={r[10]:r for r in rr}
    assert len(by)==653 and {r[0] for r in rr}=={'mewati_jhambaus','mewati_akera','Khandesi','P','kaithal','bagri_fatehabad'}
    regional=[r for r in rr if 'Bawal-Kot-Kasam' in r[10]]
    assert len(regional)==4
    for r in regional:
        tags=r[14].split()
        assert len([t for t in tags if t.startswith('dialect:')])==2
        assert any('bawal_nabha' in t for t in tags) and any('kot_kasam_jaipur' in t for t in tags)
        assert not any('gurgaon' in t for t in tags)
    assert sum(bool(r[11]) for r in rr)==12
    component=by['lsi9.2:p51:Panjabi-comparison-whole-copulas:w3']
    assert component[2]=='sai-gō'
    assert component[13].split('|')==['lsi9.2:p51:present-copula:p3:sg:w1','lsi9.2:p51:Panjabi-comparison-whole-copulas:w1']
    assert [by[k][2] for k in component[13].split('|')]==['sai','gō']
    for key in ['lsi9.2:p51:Panjabi-comparison-whole-copulas:w1','lsi9.2:p51:Panjabi-comparison-whole-copulas:w2']:
        assert 'pres' not in by[key][14].split() and {'affix','suffix'}<=set(by[key][14].split())

def test_review_metadata_does_not_add_english_only_noun_number_or_passive_voice():
    by={r[10]:r for r in rows(FINAL/'combined-review-candidate.csv')}
    sons=by['lsi9.2:p234:aligned-line:1:word:4']
    assert (sons[2],sons[3])==('bēṭā','sons')
    assert 'pl' not in sons[14].split() and {'noun','kinship'}<=set(sons[14].split())
    kaha=by['lsi9.2:p238:aligned-line:10:word:3']
    assert (kaha[2],kaha[3])==('kahā','it-was-said') and 'pass' not in kaha[14].split()
    audit={a['source_unit_key']:a for a in records(FINAL/'combined-review-audit.jsonl')}
    packet=audit[kaha[10]]['source_record']['root_metadata_review']['exact_keyed_decision']
    assert any(h['type']=='voice-analysis-scope' for h in packet['additional_scope_holds'])
    assert audit[kaha[10]]['source_record']['typed_metadata_uncertainties']==packet['additional_scope_holds']

def test_broad_native_provenance_is_complete_without_narrow_assignment():
    audit=records(FINAL/'combined-review-audit.jsonl')
    native=[a['source_record'] for a in audit if a['kind']=='No38-physical-Native']
    spans=[a['source_record'] for a in audit if a['kind']=='No38-broad-script-provenance']
    assert len(native)==194 and len(spans)==8
    assert Counter(k for s in spans for k in s['native_keys'])==Counter(r['key'] for r in native)
    corrected=next(r for r in native if r['key']=='lsi9.2:p237:native-line:4:word:15')
    assert corrected['raw_native']=='گھروالو' and corrected['historical_source_record']['raw_native']=='گھروالے'
    assert sum(bool(r['typed_uncertainties']) for r in native)==3
    assert sum(r['raw_native']=='—' for r in native)==29
    assert sum(r['raw_native']=='۲۱' for r in native)==1
    narrow=[a['actual_row'] for a in audit if a['kind'] in ('No38-atomic','No38-expression')]
    assert len(narrow)==155 and all(r[4]==r[5]=='' for r in narrow)
    docs=[a for a in audit if a['kind']=='documentary-source-block']
    assert len(docs)==5
    intro=next(a for a in docs if 'two-specimen-script-and-lect-introduction' in a['source_unit_key'])
    assert intro['source_record']['raw_physical_line_reading'] and intro['entry_keys']==[]

def test_review_preserves_source_fields_and_all_physical_dispositions():
    old=rows(HERE/'review-assembly-r1/combined-review-candidate.csv');new=rows(FINAL/'combined-review-candidate.csv')
    assert len(old)==len(new)==653
    assert all(all(a[i]==b[i] for i in (1,2,3,4,5,8,9,10,12)) for a,b in zip(old,new))
    audit=records(FINAL/'combined-review-audit.jsonl')
    assert len(audit)==len({a['source_unit_key'] for a in audit})==1970
    assert sum(a['kind']=='structural-parent' for a in audit)==173
    assert sum(a['kind']=='grammar-documentary' for a in audit)==23
    physical=[a for a in audit if a['kind'] in ('No37-physical-Roman','No37-physical-Native','No38-physical-Roman','No38-physical-Native')]
    assert len(physical)==len({a['physical_key'] for a in physical})==1059
    assert {k for a in audit for k in a['entry_keys']}=={r[10] for r in new}
    assert all(a['accepted_for_installation'] is False for a in audit)
