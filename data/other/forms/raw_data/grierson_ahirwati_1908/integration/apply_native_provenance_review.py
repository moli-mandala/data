"""Guarded documentary Native derivative; never assigns broad spans to lexical Native."""
import copy,hashlib,json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
OUT=HERE/'specimen38-native-provenance-r1'
APPROVAL='specimen38/independent-native-review/root-native-provenance-acceptance.json'
APPROVAL_SHA='0ca9166f71a86d5983a32dd5dfedff45118884f85fc47e09000c1674c36ba4d3'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def records(p):return [json.loads(s) for s in p.read_text().splitlines()]
def save(p,content):
    if p.exists():assert p.read_text()==content,('Immutable derivative differs',str(p))
    else:p.write_text(content)

def main():
    approval_path=PACKAGE/APPROVAL
    assert sha(approval_path)==APPROVAL_SHA
    approval=json.loads(approval_path.read_text());assert approval['status']=='pass'
    pins={APPROVAL:APPROVAL_SHA}
    for name,h in approval['inputs'].items():
        rel='specimen38/'+name;assert sha(PACKAGE/rel)==h,rel;pins[rel]=h
    def read(rel):
        pins[rel]=sha(PACKAGE/rel);return records(PACKAGE/rel)
    raw=read('specimen38/native-review/native-atoms-first-reading.jsonl')
    decisions=read('specimen38/independent-native-review/all-194-keyed-native-decisions.jsonl')
    spans=read('specimen38/native-review/native-roman-span-alignment-proposal.jsonl')
    span_review=read('specimen38/independent-native-review/all-8-broad-script-span-decisions.jsonl')
    roman=read('specimen38/root-reviewed-revision/p238-atomic-reviewed.jsonl')
    roman_by={r['source_unit_key']:r for r in roman};assert len(roman_by)==133
    census_status_path=PACKAGE/'scope/root-current-census-reconciliation-r2-status.json'
    census_status=json.loads(census_status_path.read_text())
    pins[str(census_status_path.relative_to(PACKAGE))]=sha(census_status_path)
    for name,h in census_status['inputs'].items():assert sha(PACKAGE/name)==h;pins[name]=h
    census=read('scope/root-current-census-reconciliation-r2.jsonl')
    assert pins['scope/root-current-census-reconciliation-r2.jsonl']==census_status['output_sha256']
    fine_keys=[k for r in census for k in r['fine_source_keys']]
    assert len(census)==173
    # The complete census also carries grammar/comparison children; 1059 is
    # specifically the two specimens' physical Roman/Native key inventory.
    physical=[]
    for rel,keyfield in [('specimens/specimen37/roman-atoms-first-reading.jsonl','key'),
                         ('specimens/specimen37/native-atoms-first-reading.jsonl','key'),
                         ('specimen38/root-reviewed-revision/p238-atomic-reviewed.jsonl','source_unit_key'),
                         ('specimen38/native-review/native-atoms-first-reading.jsonl','key')]:
        physical.extend(r[keyfield] for r in read(rel))
    assert len(physical)==len(set(physical))==1059
    assert Counter(k for k in fine_keys if k in set(physical))==Counter(physical)
    assert len(raw)==len(decisions)==194 and [r['key'] for r in raw]==[d['source_unit_key'] for d in decisions]
    approved_changes={d['source_unit_key']:d for d in approval['corrections']}
    assert set(approved_changes)=={'lsi9.2:p237:native-line:4:word:15'}
    revised=[]
    for r,d in zip(raw,decisions):
        assert (r['key'],r['raw_native'],r['physical_line'],r['word_rank'])==(d['source_unit_key'],d['old_native_guard'],d['physical_line'],d['word_rank'])
        assert r['key'] in fine_keys
        if d['proposed_native']!=r['raw_native']:
            assert d==approved_changes[r['key']]
        x=copy.deepcopy(r)
        x['historical_source_record']=copy.deepcopy(r)
        x['raw_native']=d['proposed_native']
        x['typed_uncertainties']=copy.deepcopy(d['current_typed_holds'])
        x['source_stage_provenance_review']=copy.deepcopy(d)
        x['root_acceptance']={'path':APPROVAL,'sha256':APPROVAL_SHA}
        x['stage']='Qualified original-backed Native physical provenance; not narrow lexical equivalence'
        revised.append(x)
    assert sum(a['raw_native']!=b['raw_native'] for a,b in zip(raw,revised))==1
    assert sum(bool(r['typed_uncertainties']) for r in revised)==3
    assert Counter(d['unit_role'] for d in decisions)==Counter({'native-script-physical-space-group':164,'separator-glyph':29,'redundant-printed-numeral':1})
    by={r['key']:r for r in revised}
    assert len(spans)==len(span_review)==8
    reviewed_spans=[]
    for r,d in zip(spans,span_review):
        assert r['key']==d['span_key'] and r['native_keys']==d['native_keys'] and r['roman_keys']==d['roman_keys']
        assert r['raw_native_span']==d['old_raw_native_guard']
        assert r['raw_roman_span']==d['historical_roman_span_guard']
        assert r['printed_interlinear_analysis']==d['historical_interlinear_guard']
        native=' '.join(by[k]['raw_native'] for k in d['native_keys'])
        current_roman=' '.join(roman_by[k]['raw_form'] for k in d['roman_keys'])
        current_gloss=' '.join(roman_by[k]['raw_gloss'] for k in d['roman_keys'])
        assert (native,current_roman,current_gloss)==(d['current_native_proposal_with_keyed_correction'],d['reviewed_roman_physical_span'],d['reviewed_interlinear_span'])
        x=copy.deepcopy(r)
        x['historical_source_record']=copy.deepcopy(r)
        x['raw_native_span']=native;x['raw_roman_span']=current_roman;x['printed_interlinear_analysis']=current_gloss
        x['current_native_hold_keys']=d['current_native_hold_keys']
        x['source_stage_provenance_review']=copy.deepcopy(d)
        x['root_acceptance']={'path':APPROVAL,'sha256':APPROVAL_SHA}
        x['status']='Accepted qualified broad provenance span; no narrow word equivalence or manufactured sentence boundary'
        reviewed_spans.append(x)
    assert Counter(k for r in reviewed_spans for k in r['native_keys'])==Counter(by.keys())
    assert Counter(k for r in reviewed_spans for k in r['roman_keys'])==Counter(roman_by.keys())
    OUT.mkdir(exist_ok=True)
    for name,items in [('native-atoms-reviewed.jsonl',revised),('broad-script-spans-reviewed.jsonl',reviewed_spans)]:
        save(OUT/name,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
    pins[str(Path(__file__).relative_to(PACKAGE))]=sha(Path(__file__))
    result={'status':'pass','inputs':pins,'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.name!='status.json'},
      'counts':{'Native_groups':194,'literal_corrections':1,'current_Native_holds':3,'horizontal_separators':29,'documentary_extra_numeral':1,'broad_spans':8,'conserved_Roman_keys':133,'census_parents':173,'census_fine_keys':1059},
      'Roman_span_refreshes':sum(a['raw_roman_span']!=b['raw_roman_span'] for a,b in zip(spans,reviewed_spans)),
      'narrow_Native_assignment':False,'canonical_changes':False,'installation_approved':False,
      'policy':'Native physical grouping and broad provenance accepted with three explicit glyph holds. Historical source records retained verbatim. No broad Native string is assigned to an atomic or whole-expression lexical Native field.'}
    save(OUT/'status.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
