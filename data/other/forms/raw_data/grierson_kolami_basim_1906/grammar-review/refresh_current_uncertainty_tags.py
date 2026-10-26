"""Current metadata derivative after separately preserving resolved cautions."""
import copy,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE/'sharp-reviewed-revision';NEW=HERE/'sharp-reviewed-revision2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def guard(r):return {f:r[f] for f in ['form','gloss','language_label','printed_label']}
def main():
    p=NEW/'reviewed-child-witnesses.jsonl'
    assert sha(p)=='c17d355aa870971bbb2433a7f9115fa5ce6deeac3ddc34911451d42534393814'
    current={r['source_unit_key']:r for r in rows(p)}
    old={r['source_unit_key']:r for r in rows(OLD/'reviewed-child-witnesses.jsonl')}
    ts=[];gs=[];decisions=[];by={}
    for before in rows(OLD/'grammar-tag-proposal.jsonl'):
        k=before['witness_key'];assert before['exact_before_guard']==guard(old[k])
        t=copy.deepcopy(before);t['prior_sharp_metadata_proposal']=before
        t['exact_before_guard']=guard(current[k])
        t['original_literal_uncertainties_retained']=current[k]['typed_uncertainties']
        t['proposed_tags']=[v for v in t['proposed_tags'] if v!='uncertain']
        if t['typed_metadata_uncertainties'] or current[k]['typed_uncertainties']:t['proposed_tags'].append('uncertain')
        if t['proposed_tags']!=before['proposed_tags']:
            decisions.append({'witness_key':k,'old_tags':before['proposed_tags'],'new_tags':t['proposed_tags'],
                              'reason':'Recomputed uncertainty union from current literal holds and retained source-analysis uncertainties; resolved reading cautions remain historical.'})
        ts.append(t);by[k]=t
    for before in rows(OLD/'grammar-graph-proposal.jsonl'):
        k=before['witness_key'];assert before['exact_before_guard']==guard(old[k])
        g=copy.deepcopy(before);g['prior_sharp_metadata_proposal']=before
        g['exact_before_guard']=guard(current[k]);g['proposed_tags_guard']=by[k]['proposed_tags']
        gs.append(g)
    assert len(ts)==224 and len(gs)==213
    assert sum(bool(g['variant_of_witness_key']) for g in gs)==15 and sum(bool(g['ordered_parents']) for g in gs)==2
    for name,rr in [('grammar-tag-proposal.jsonl',ts),('grammar-graph-proposal.jsonl',gs)]:
        (NEW/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
    report={'status':'Current uncertainty recomputed without deleting historical or source-analysis evidence',
      'input_pins':{str(x):sha(x) for x in [p,OLD/'grammar-tag-proposal.jsonl',OLD/'grammar-graph-proposal.jsonl']},
      'counts':{'tag_records':224,'current_literal_uncertain':7,'uncertain_tagged':sum('uncertain' in t['proposed_tags'] for t in ts),'graph_records':213,'variant_pointers':15,'ordered_component_sets':2},
      'exact_tag_deltas':decisions,'all_nonuncertain_tags_and_graph_claims_preserved':True,
      'outputs':{n:sha(NEW/n) for n in ['grammar-tag-proposal.jsonl','grammar-graph-proposal.jsonl']},'canonical_changes':False}
    (NEW/'metadata-guard-refresh.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['counts']))
if __name__=='__main__':main()
