"""Apply exact independent/root-corroborated p49–51 corrections, keeping history."""
import copy
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[7]
HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent/'independent-grammar-review/independent-full-grammar-review.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,rows):
    raw=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows).encode()
    if p.exists() and p.read_bytes()!=raw: raise SystemExit('Refusing historical overwrite '+str(p))
    p.write_bytes(raw)
def main():
    review=json.loads(REVIEW.read_text())
    for path,h in review['inputs'].items(): assert sha(ROOT/path)==h,(path,h)
    old=[json.loads(x) for x in (HERE/'child-witnesses-first-reading.jsonl').read_text().splitlines()]
    rows=copy.deepcopy(old); index={r['witness_key']:r for r in rows}; changes=[]
    for change in review['corrections']:
        r=index[change['key']]; assert r[change['field']]==change['old']; assert r['source_gloss']==''
        r[change['field']]=change['new']; changes.append(change)
    parent='lsi9.2:p50:nominal-ending-comparison-continuation'
    contexts={parent:'Strong masculine a-base: nominative singular ō, oblique ā; thus agreeing with Mewati. Bangaru nominative ā, oblique ē. Same rule in adjectives and genitive suffixes, continuing p51.',
              'lsi9.2:p51:qualified-oblique-relative-interrogative':'Probably oblique singular of jō and kauṇ are jaĩh or jā̃h and kaĩh or kā̃h respectively, but I have not come across examples.'}
    for r in rows:
        if r['parent_key'] in contexts:
            changes.append({'key':r['witness_key'],'field':'source_quote','old':r['source_quote'],'new':contexts[r['parent_key']], 'reason':'Scoped context-summary correction, not a new quoted occurrence.'})
            r['source_quote']=contexts[r['parent_key']]
        if r['witness_key'] in review['resolved_transcription_holds']:
            r['literal_review_resolution']=review['resolved_transcription_holds'][r['witness_key']]
            r['historical_typed_uncertainties']=r['typed_uncertainties']; r['typed_uncertainties']=[]
        r['stage']='reviewed literal derivative; final emission/config audit pending'
    additions=[]
    for n,claim in enumerate(review['completeness_recommendations']):
        sourcekey=claim['parent_key']+(':w2' if n in (0,2) else ':w3')
        if n==2: sourcekey=claim['parent_key']+':w1'
        newkey=claim['parent_key']+(':w6' if n==0 else ':w7' if n==1 else ':w2')
        source=index[sourcekey]; assert source['source_form']==claim['source_form']; assert source['source_gloss']==''
        added=copy.deepcopy(source); added.update(witness_key=newkey,language_label='Mewati',
             disposition='eligible_comparator',source_analysis=claim['source_analysis'],
             physical_source_keys=[sourcekey],physical_span_shared=True,
             qualification=claim['source_evidence']+'; same printed token, additional explicit language interpretation.',typed_uncertainties=[])
        additions.append(added); rows.append(added)
    assert {r['witness_key'] for r in old}.issubset({r['witness_key'] for r in rows})
    assert len(rows)==98 and len({r['witness_key'] for r in rows})==98
    parents=[json.loads(x) for x in (HERE/'parent-coverage-first-reading.jsonl').read_text().splitlines()]
    for p in parents:
        if p['key'] in contexts: p['historical_source_context']=p['source_context']; p['source_context']=contexts[p['key']]
        p['child_keys']=[r['witness_key'] for r in rows if r['parent_key']==p['key']]
    dump(HERE/'child-witnesses-reviewed.jsonl',rows); dump(HERE/'parent-coverage-reviewed.jsonl',parents)
    report={'status':'all95 first-reading keys preserved; four field corrections plus three explicit shared-token interpretations',
            'first_reading_sha256':sha(HERE/'child-witnesses-first-reading.jsonl'),'independent_review_sha256':sha(REVIEW),
            'script_sha256':sha(Path(__file__)),'changes':changes,'added_interpretations':additions,
            'outputs':{f:sha(HERE/f) for f in ['child-witnesses-reviewed.jsonl','parent-coverage-reviewed.jsonl']},
            'counts':{'parents':len(parents),'interpretations':len(rows),'new_physical_tokens':0},
            'notes':'No graph edges, canonical files, unsupported English glosses or phonemic normalization. Root final derivative approval pending.'}
    p=HERE/'reviewed-derivative-continuity.json'; raw=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
    if p.exists() and p.read_bytes()!=raw: raise SystemExit('Refusing historical overwrite')
    p.write_bytes(raw); print(json.dumps({'rows':len(rows),'parents':len(parents),'output_sha256':report['outputs']}))
if __name__=='__main__': main()
