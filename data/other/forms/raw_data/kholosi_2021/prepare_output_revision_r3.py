"""New borrowing-class and historical-reference revision; immutable r2 retained."""
import copy
import csv
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'graph-reconciliation-r3'
old=list(csv.reader((P/'whole-proposal-r2/whole-proposed.csv').open()))
rows=copy.deepcopy(old);bykey={r[10]:r for r in rows}
donors=json.loads((R/'donor-classification-proposal.json').read_text())
historical=json.loads((R/'historical-assignment-correction-proposal.json').read_text())
for key,evidence in donors['row_classifications'].items():
 row=bykey[key];tags=row[14].split()+['loanword']
 if any(e['qualified'] for e in evidence):tags.append('uncertain')
 row[14]=' '.join(dict.fromkeys(tags))
for c in historical['corrections']:
 row=bykey[c['entry_key']]
 assert row[1]==c['old_parameter']
 row[1]=c['proposed_parameter']
 before=f"Source ancestry claim differs from retained historical Jambu assignment {c['old_parameter']}: "
 after=f"Earlier Jambu assignment {c['old_parameter']} was corrected to {c['proposed_parameter']} based on the source's named head. Source claim: "
 if before in row[6]:row[6]=row[6].replace(before,after)
 else:
  row[6]=' '.join(filter(None,(row[6],c['reason'])))
 if not c['proposed_parameter']:
  row[14]=' '.join(dict.fromkeys(row[14].split()+['uncertain']))
delta=[]
for a,b in zip(old,rows):
 assert all(a[i]==b[i] for i in (0,2,3,4,5,7,8,9,10,11,12,13))
 if a!=b:delta.append(dict(entry_key=b[10],changes={str(i):dict(before=a[i],after=b[i]) for i in range(15) if a[i]!=b[i]}))
with (R/'whole-proposed.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
(R/'r2-to-r3-delta.json').write_text(json.dumps(delta,ensure_ascii=False,indent=2)+'\n')
summary=dict(status='proposal; independent audit pending',rows=len(rows),changed_rows=len(delta),
 historical_assignment_changes=[c['entry_key'] for c in historical['corrections']],
 borrowing_classified_rows=len(donors['row_classifications']),all_keys_preserved=True,
 lexical_fields_unchanged=True,proposal_sha256=hashlib.sha256((R/'whole-proposed.csv').read_bytes()).hexdigest())
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
