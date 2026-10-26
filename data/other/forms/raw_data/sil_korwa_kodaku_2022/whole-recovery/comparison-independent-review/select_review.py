from pathlib import Path
import hashlib,json,random
H=Path(__file__).resolve().parent; B=H.parent/'comparisons'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(l) for l in (B/'comparison-reviewed-candidate.jsonl').read_text().splitlines()]
rng=random.Random(20260927501); selected=[]; pages=set()
for code in ['A','B','J','T','U','V','k']:
    pool=[r for r in rows if r['source_code']==code and not r.get('field_decision') and ':physical208' not in r['source_unit_key']]
    for i in range(2 if code=='k' else 3):
        fresh=[r for r in pool if r['historical_audit_record']['PDF_Page'] not in pages]
        r=rng.choice(fresh or pool); selected.append(r); pool.remove(r); pages.add(r['historical_audit_record']['PDF_Page'])
controls=[r for r in rows if r.get('field_decision') or ':physical208' in r['source_unit_key'] or (r['item'],r['source_code']) in [(17,'k'),(33,'A'),(33,'J'),(185,'A')]]
result=dict(seed=20260927501,method='Seven source-code strata; 20 distinct pages where available; all proposed correction classes independently targeted separately.',
    pins={n:sha(B/n) for n in ['comparison-reviewed-candidate.jsonl','comparison-review-accounting.jsonl','owner-original-review.json','owner-field-corrections.json','prepare_reviewed_candidate.py']},
    selected_rows=selected,targeted_rows=controls)
out=H/'selection.json'
if out.exists():assert json.loads(out.read_text())==result
else:out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for i,r in enumerate(selected,1):print(i,r['historical_audit_record']['PDF_Page'],r['item'],r['source_code'],r['form'],r['gloss'],r['tags'])
print('CONTROLS')
for r in controls:print(r['historical_audit_record']['PDF_Page'],r['item'],r['source_code'],r['form'],r['gloss'],r['tags'])
