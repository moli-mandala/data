import hashlib,json,random
from pathlib import Path
H=Path(__file__).resolve().parent;O=H.parent/'interlinear-review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(x) for x in (O/'continuous-semantic-proposal.jsonl').read_text().splitlines()]
inputs=json.loads((O/'continuous-review-input.json').read_text())
rng=random.Random(20260927503);selection=[]
for section,n in [('C.2',7),('C.3',6),('C.4',7)]:
    pool=[r for r in rows if r['section']==section];used=set()
    for _ in range(n):
        choices=[r for r in pool if r['physical_p_key'] not in used]
        r=rng.choice(choices);selection.append(r);used.add(r['physical_p_key'])
pins={f:sha(O/f) for f in ['continuous-semantic-proposal.jsonl','continuous-semantic-coverage.json','continuous-review-input.json','continuous_alignment_specs.py','prepare_continuous_alignment.py']}
(H/'selection.json').write_text(json.dumps(dict(seed=20260927503,input_hashes=pins,selected_rows=selection),ensure_ascii=False,indent=2)+'\n')
for i,r in enumerate(selection,1):print(i,r['source_unit_key'],r['form'],'=',r['gloss'],'RAW',r['raw_interlinear_gloss'],r['tags'])
coverage=[]
for x in inputs:
    rr=[r for r in rows if r['physical_p_key']==x['key']]
    assert ' '.join(r['form'] for r in rr)==' '.join(x['form_tokens'])
    assert ' '.join(r['raw_interlinear_gloss'] for r in rr)==x['gloss']
    at=1
    for r in rr:
        assert r['target_token_range'][0]==at
        at=r['target_token_range'][1]+1
    assert at==len(x['form_tokens'])+1
    coverage.append(dict(index=x['index'],key=x['key'],children=len(rr),target_tokens=len(x['form_tokens']),conservation='pass',pairs=[dict(key=r['source_unit_key'],form=r['form'],gloss=r['gloss'],raw_gloss=r['raw_interlinear_gloss']) for r in rr]))
(H/'partition-conservation.json').write_text(json.dumps(dict(input_hashes=pins,blocks=coverage),ensure_ascii=False,indent=2)+'\n')
