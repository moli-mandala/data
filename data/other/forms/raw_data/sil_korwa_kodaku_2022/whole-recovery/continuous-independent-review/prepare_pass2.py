import hashlib,json,random,sys
from pathlib import Path
H=Path(__file__).resolve().parent; O=H.parent/'interlinear-review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert len(sys.argv)==2 and sha(O/'continuous-semantic-proposal.jsonl')==sys.argv[1]
first=json.loads((H/'independent-continuous-review.json').read_text())
rows=[json.loads(s) for s in (O/'continuous-semantic-proposal.jsonl').read_text().splitlines()]
bykey={r['source_unit_key']:r for r in rows}
checks=[]
for c in first['keyed_recommendations']:
    r=bykey[c['source_unit_key']]
    checks.append(dict(key=c['source_unit_key'],recommended_tags=c['proposed'],actual_tags=r['tags'],all_recommended_present=set(c['proposed'])<=set(r['tags']),literal_continuity=r['form']==c['form'] and r['raw_interlinear_gloss']==c['raw_interlinear_gloss']))
assert all(c['all_recommended_present'] and c['literal_continuity'] for c in checks)
excluded={s['actual_proposal']['physical_p_key'] for s in first['sample']}
rng=random.Random(20260927504); chosen=[];used=set(excluded)
for section,n in [('C.2',7),('C.3',6),('C.4',7)]:
    for i in range(n):
        pool=[r for r in rows if r['section']==section and r['physical_p_key'] not in used]
        r=rng.choice(pool); chosen.append(r);used.add(r['physical_p_key'])
pins={p.name:sha(p) for p in [O/'continuous-semantic-proposal.jsonl',O/'continuous-semantic-coverage.json',O/'continuous-review-input.json',O/'continuous_alignment_specs.py',O/'prepare_continuous_alignment.py']}
out=dict(seed=20260927504,input_hashes=pins,previous_report_sha256=sha(H/'independent-continuous-review.json'),excluded_physical_blocks=sorted(excluded),class_checks=checks,selected_rows=chosen)
dest=H/'pass2-selection.json'
assert not dest.exists(), 'Selection immutable; do not redraw'
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for i,r in enumerate(chosen,1):print(i,r['source_unit_key'],r['form'],'=',r['gloss'],'RAW',r['raw_interlinear_gloss'],r['tags'])
