"""Exact keyed independent recommendations; no free-text grammar inference."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
P=H.parent/'continuous-independent-review/independent-continuous-review.json'
assert hashlib.sha256(P.read_bytes()).hexdigest()=='38da87e471a3530bbb4e55cddb9f22ceb11511883144edb172d9686ace010b86'
REPORT=json.loads(P.read_text())
BASE=H/'revision1/continuous-semantic-proposal.jsonl'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==REPORT['input_hashes']['continuous-semantic-proposal.jsonl']
OLD={r['source_unit_key']:r for r in map(json.loads,BASE.read_text().splitlines())}
RECOMMENDATIONS={r['source_unit_key']:r for r in REPORT['keyed_recommendations']}
assert len(RECOMMENDATIONS)==90
for key,r in RECOMMENDATIONS.items():
 assert r['field']=='tags' and OLD[key]['tags']==r['old']
 assert OLD[key]['form']==r['form'] and OLD[key]['raw_interlinear_gloss']==r['raw_interlinear_gloss']

def apply(rows):
 seen=set()
 for row in rows:
  key=row['source_unit_key']
  if key not in RECOMMENDATIONS:continue
  r=RECOMMENDATIONS[key]
  assert row['form']==r['form'] and row['raw_interlinear_gloss']==r['raw_interlinear_gloss']
  assert set(row['tags'])-set(r['old'])<={'uncertain'}, ('Unreviewed intervening tag change',key)
  row['tags']=list(dict.fromkeys([*row['tags'],*r['proposed']]))
  row['grammar_review']={'report':str(P.relative_to(H.parent)),'category':r['category'],'qualification':r['qualification']}
  seen.add(key)
 assert seen==set(RECOMMENDATIONS)
