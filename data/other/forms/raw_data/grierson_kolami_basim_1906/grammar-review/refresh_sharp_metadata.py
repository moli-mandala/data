"""Explicitly rebase reviewed metadata guards through the sharp literal overlay."""
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
OUT=HERE/'sharp-reviewed-revision'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p): return [json.loads(s) for s in p.read_text().splitlines()]
def guard(r): return {f:r[f] for f in ['form','gloss','language_label','printed_label']}
def write(p,rs): p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
def main():
    oldpath=HERE/'reviewed-child-witnesses.jsonl';newpath=OUT/'reviewed-child-witnesses.jsonl'
    tagpath=PACKAGE/'independent-grammar-review/independent-grammatical-tag-proposal.jsonl'
    graphpath=PACKAGE/'independent-grammar-review/independent-grammar-graph-proposal.jsonl'
    pins={oldpath:'ce78a459c1102d2b6e6376ea953f23dae4ebf9a5888daa63c9c72eae38c59ed2',
          newpath:'3b0621fd4ed10d80758ec9f448bd4e49a466be25fd89e45ed412a568223d1b13',
          tagpath:'8281961b38c5fac7220841f8713aad67f8db25101f78208a22d5d7524e7ecc0f',
          graphpath:'6cdfb17c5c960ad04e63a0f3257ace84125b7d03e3e24f72cdb65e3343998f1e'}
    for p,pin in pins.items(): assert sha(p)==pin,(p,sha(p))
    old={r['source_unit_key']:r for r in rows(oldpath)}
    new={r['source_unit_key']:r for r in rows(newpath)}
    assert list(old)==list(new) and len(old)==224
    tags=[];taglookup={};graphs=[]
    for t0 in rows(tagpath):
        k=t0['witness_key'];assert t0['exact_before_guard']==guard(old[k]),k
        t=copy.deepcopy(t0);t['historical_proposal']=t0;t['exact_before_guard']=guard(new[k])
        t['original_literal_uncertainties_retained']=new[k]['typed_uncertainties']
        if k=='grierson1906kolamibasim:p566:present-copula:3:independent:w1':
            assert old[k]['printed_label']=='Sing. 2. m. present verb substantive'
            assert new[k]['printed_label']=='Sing. 3. m. present verb substantive'
            t['typed_metadata_uncertainties']=[u for u in t['typed_metadata_uncertainties'] if u['type']!='source-numbering']
            t['proposed_tags']=[v for v in t['proposed_tags'] if v!='uncertain']+['3sg','third-person']
            t['source_scope_rationale']='Sharper original unequivocally prints Sing.3.m.; historical degraded 2.m. retained in audit only.'
        if new[k]['typed_uncertainties'] or t['typed_metadata_uncertainties']:
            if 'uncertain' not in t['proposed_tags']:t['proposed_tags'].append('uncertain')
        t['guard_rebase']='Exact immutable historical guard checked, then mapped to same sharp source key; no silent old-input acceptance.'
        tags.append(t);taglookup[k]=t
    for g0 in rows(graphpath):
        k=g0['witness_key'];assert g0['exact_before_guard']==guard(old[k]),k
        assert g0['proposed_tags_guard']==next(t['proposed_tags'] for t in rows(tagpath) if t['witness_key']==k)
        g=copy.deepcopy(g0);g['historical_proposal']=g0
        g['exact_before_guard']=guard(new[k]);g['proposed_tags_guard']=taglookup[k]['proposed_tags']
        for e in g.get('endpoint_guards',[]):
            ek=e.get('witness_key',e.get('key'))
            assert ek in old,(k,e)
            for f in ['form','gloss','language_label','printed_label']:
                if f in e:
                    assert e[f]==old[ek][f],(k,ek,f,e[f],old[ek][f])
                    e[f]=new[ek][f]
        g['guard_rebase']='Same exact keys and source claims; only guards refreshed through owner-corroborated sharp derivative.'
        graphs.append(g)
    assert len(tags)==224 and len(graphs)==213
    assert sum(bool(g['variant_of_witness_key']) for g in graphs)==15
    assert sum(bool(g['ordered_parents']) for g in graphs)==2
    write(OUT/'grammar-tag-proposal.jsonl',tags)
    write(OUT/'grammar-graph-proposal.jsonl',graphs)
    report={'status':'Guard refresh through exact reviewed source keys; final combined-output acceptance pending',
        'input_pins':{str(p.relative_to(PACKAGE)):v for p,v in pins.items()},'counts':{'tag_records':224,'graph_records':213,'variant_pointers':15,'ordered_component_sets':2},
        'only_semantic_tag_change':'p566 printed3.m. now gets3sg/third-person and historical source-numbering uncertainty removed; other uncertainty flags union-preserved.',
        'outputs':{p.name:sha(p) for p in [OUT/'grammar-tag-proposal.jsonl',OUT/'grammar-graph-proposal.jsonl']},'canonical_changes':False}
    (OUT/'metadata-guard-refresh.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['counts']))
if __name__=='__main__':main()
