"""Freeze graph reconciliation separately from immutable r1."""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'graph-reconciliation-r2'
F=P/'whole-proposal-r2'
assert not F.exists(),'Refuse to overwrite an existing independent-review freeze.'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
scripts=['prepare_graph_claim_review_r2.py','review_numeric_claims_r2.py','prepare_local_components_r2.py',
 'review_local_components_r2.py','complete_claim_census_r2.py','prepare_graph_output_draft_r2.py']
before={p.name:sha(p) for p in R.iterdir() if p.is_file()}
for name in scripts:subprocess.run([sys.executable,str(P/name)],check=True)
assert all(sha(R/name)==digest for name,digest in before.items()),'Regeneration changed reviewed draft.'
report=json.loads((R/'actual-parser-validation.json').read_text())
assert report['proposal_sha256']==sha(R/'whole-graph-draft.csv')
rows=list(csv.reader((R/'whole-graph-draft.csv').open()));bykey={r[10]:r for r in rows}
old=list(csv.reader((P/'whole-proposal-r1/whole-proposed.csv').open()))
claims={r['source_unit_key']:r for r in map(json.loads,(R/'complete-source-claim-dispositions.jsonl').open())}
actions=list(map(json.loads,(R/'row-graph-actions.jsonl').open()))
audit=list(map(json.loads,(P/'whole-proposal-r1/whole-proposed-audit.jsonl').open()))
for u in audit:
 key=u['source_unit_key']
 if key in claims:u['graph_claim_review_r2']=claims[key]
 u['actual_emitted_rows_r2']=[bykey[k] for k in u['emitted_entry_keys']]
 u['graph_actions_r2']=[a for a in actions if a.get('entry_key') in u['emitted_entry_keys']]
F.mkdir()
shutil.copyfile(R/'whole-graph-draft.csv',F/'whole-proposed.csv')
for name in ('whole-profile-proposed.txt','whole-metadata-proposed.yaml'):
 shutil.copyfile(P/'whole-proposal-r1'/name,F/name)
(F/'whole-proposed-audit.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in audit))
delta=[dict(entry_key=b[10],changes={str(i):dict(before=a[i],after=b[i]) for i in range(15) if a[i]!=b[i]}) for a,b in zip(old,rows) if a!=b]
(F/'r1-to-r2-delta.json').write_text(json.dumps(delta,ensure_ascii=False,indent=2)+'\n')
shutil.copyfile(R/'actual-parser-validation.json',F/'whole-parser-validation.json')
manifest=dict(status='frozen revision2; fresh independent acceptance required',rows=len(rows),audit_units=len(audit),
 legacy_keys_preserved=198,legacy_assignments_preserved=198,changed_rows=len(delta),
 input_sha256={str(p.relative_to(P)):sha(p) for p in [
  P/'whole-proposal-r1/review-manifest.json',P/'whole-proposal-r1/whole-proposed.csv',
  P/'whole-proposal-r1/whole-proposed-audit.jsonl',P/'source-etymology-claims.jsonl',
  *sorted(p for p in R.iterdir() if p.is_file()),*[P/s for s in scripts],
  P/'validate_graph_draft_r2.py',Path(__file__)]},
 output_sha256={p.name:sha(p) for p in sorted(F.iterdir()) if p.is_file()},
 method='Immutable r1 plus explicit source-claim graph overlay; all lexical fields, IPA, source citations and keys remain exact. No canonical installation.',
 deferred=['Fresh independent r2 output/source audit','Canonical installation','Compiled graph and identity','Full build/test suite','Browser QA'])
(F/'review-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False))
