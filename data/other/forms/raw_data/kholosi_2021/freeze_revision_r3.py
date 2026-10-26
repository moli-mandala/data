"""Freeze the independently reviewed donor-class repair for fresh output audit."""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P/'graph-reconciliation-r3';F=P/'whole-proposal-r3'
assert not F.exists(),'Never overwrite a review freeze.'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={p.name:sha(p) for p in R.iterdir() if p.is_file()}
scripts=['prepare_historical_assignment_review_r3.py','prepare_donor_classification_r3.py','prepare_output_revision_r3.py']
for script in scripts:subprocess.run([sys.executable,str(P/script)],check=True)
assert all(sha(R/n)==h for n,h in before.items()),'Regeneration drift.'
report=json.loads((R/'actual-parser-validation.json').read_text())
assert report['proposal_sha256']==sha(R/'whole-proposed.csv')
rows={r[10]:r for r in csv.reader((R/'whole-proposed.csv').open())}
donors=json.loads((R/'donor-classification-proposal.json').read_text())
history=json.loads((R/'historical-assignment-correction-proposal.json').read_text())
corrections={c['entry_key']:c for c in history['corrections']}
byclaim={c['source_unit_key']:c for c in donors['claims']}
audit=list(map(json.loads,(P/'whole-proposal-r2/whole-proposed-audit.jsonl').open()))
for u in audit:
 u['actual_emitted_rows_r3']=[rows[k] for k in u['emitted_entry_keys']]
 if u['source_unit_key'] in byclaim:u['donor_claim_review_r3']=byclaim[u['source_unit_key']]
 u['row_borrowing_classification_r3']={k:donors['row_classifications'][k] for k in u['emitted_entry_keys'] if k in donors['row_classifications']}
 u['historical_assignment_corrections_r3']=[corrections[k] for k in u['emitted_entry_keys'] if k in corrections]
F.mkdir()
for name in ('whole-proposed.csv','r2-to-r3-delta.json','actual-parser-validation.json','summary.json'):
 shutil.copyfile(R/name,F/name)
for name in ('whole-profile-proposed.txt','whole-metadata-proposed.yaml'):
 shutil.copyfile(P/'whole-proposal-r2'/name,F/name)
(F/'whole-proposed-audit.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in audit))
paths=[P/'whole-proposal-r2/review-manifest.json',P/'whole-proposal-r2/whole-proposed.csv',P/'whole-proposal-r2/whole-proposed-audit.jsonl',
       P/'independent-r2-original-review.json',P/'independent-historical-link-review-r3.json',P/'independent-donor-class-pre-freeze-review-r3.json',
       *sorted(p for p in R.iterdir() if p.is_file()),*[P/s for s in scripts],P/'validate_revision_r3.py',Path(__file__)]
manifest=dict(status='frozen revision3; fresh independent acceptance required',rows=len(rows),audit_units=len(audit),
 legacy_keys_preserved=198,historical_assignment_corrections=5,other_historical_assignments_preserved=193,
 input_sha256={str(p.relative_to(P)):sha(p) for p in paths},
 output_sha256={p.name:sha(p) for p in sorted(F.iterdir()) if p.is_file()},
 method='R2 plus full donor-class reconciliation and five independently supported historical-reference corrections. All lexical fields, IPA, citations, keys and local relations remain exact.',
 deferred=['Fresh independent r3 output/source audit','Canonical installation','Compiled identity/graph','Full build/test suite','Browser QA'])
(F/'review-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest))
