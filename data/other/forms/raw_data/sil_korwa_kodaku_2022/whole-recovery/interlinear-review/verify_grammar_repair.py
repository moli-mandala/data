from pathlib import Path
import json,hashlib
from reviewed_grammar_overlay import RECOMMENDATIONS
H=Path(__file__).resolve().parent
load=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
old=load(H/'before-grammar-repair/continuous-semantic-proposal.jsonl')
new=load(H/'continuous-semantic-proposal.jsonl');assert len(old)==len(new)==790
changes=[]
for a,b in zip(old,new):
 assert a['source_unit_key']==b['source_unit_key']
 fields={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
 assert fields<={'tags','grammar_review'}
 if fields:
  assert b['source_unit_key'] in RECOMMENDATIONS
  assert set(a['tags'])<=set(b['tags'])
  assert set(RECOMMENDATIONS[b['source_unit_key']]['proposed'])<=set(b['tags'])
  changes.append(b['source_unit_key'])
assert len(changes)==90 and set(changes)==set(RECOMMENDATIONS)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'passed','rows':790,'changed_rows':90,'changed_fields':['tags','grammar_review'],'all_other_fields_identical':True,'prior_uncertainty_preserved':True,'proposal_sha256':sha(H/'continuous-semantic-proposal.jsonl'),'baseline_sha256':sha(H/'before-grammar-repair/continuous-semantic-proposal.jsonl'),'independent_review_sha256':sha(H.parent/'continuous-independent-review/independent-continuous-review.json'),'changed_keys':changes}
(H/'grammar-repair-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='changed_keys'})
