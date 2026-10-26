"""Apply reviewed graph proposals to a new draft; not an installer or freeze."""
import copy
import csv
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'graph-reconciliation-r2'
baseline=list(csv.reader((P/'whole-proposal-r1/whole-proposed.csv').open()))
rows=copy.deepcopy(baseline); bykey={r[10]:r for r in rows}
decisions=list(map(json.loads,(R/'numeric-claim-decisions.jsonl').open()))
components=json.loads((R/'local-component-decisions.json').read_text())
audit=list(map(json.loads,(P/'whole-proposal-r1/whole-proposed-audit.jsonl').open()))
audit_by_key={u['source_unit_key']:u for u in audit}
actions=[]
def note(row,text):
 if text not in row[6]:row[6]=' '.join(filter(None,(row[6],text)))
def uncertain(row):
 row[14]=' '.join(dict.fromkeys(row[14].split()+['uncertain']))
for d in decisions:
 for key in d['emitted_entry_keys']:
  row=bykey[key];ids=d['proposed_parameter_ids']
  action=dict(claim_key=d['claim_key'],entry_key=key,old_parameter=row[1],proposed_ids=ids)
  if ids:
   assert len(ids)==1
   if key.startswith('legacy:'):
    action['disposition']='historical assignment retained'
    if row[1]!=ids[0]:
     note(row,f"Source ancestry claim differs from retained historical Jambu assignment {row[1]}: {d['literal_claim']}")
     action['conflict']=True
   else:
    assert not row[1] or row[1]==ids[0]
    row[1]=ids[0];action['disposition']='source-supported ancestry assigned'
   if d['source_qualified']:
    uncertain(row)
    note(row,'The ancestry proposal is explicitly qualified in the source; its original qualification is retained in Etymology.')
   for correction in d.get('number_head_corrections',[]):
    note(row,f"The source prints CDIAL {correction['printed_id']}, whose head is {correction['printed_id_target_head']}; its explicitly named {correction['intended_head']} corresponds uniquely to CDIAL {correction['proposed_id']}. The printed number is retained as source evidence.")
  else:
   action['disposition']='no new ancestry link'
  actions.append(action)
for u in components:
 for key in u['emitted_entry_keys']:
  row=bykey[key]
  for c in u['components']:
   parents=c['accepted_entry_keys']
   if c['disposition']=='explicit alternate component set':
    assert u['source_form']=='nevalem(te)'
    parents=[p for p in parents if row[2]=='ne'+bykey[p][2]]
    assert len(parents)==1,(row,c)
   for parent in parents:
    assert parent in bykey and parent!=key
    row[13]=' '.join(dict.fromkeys(row[13].split()+[parent]))
   if parents and u['source_qualified']:
    uncertain(row)
    note(row,'The source qualifies part of this component analysis; the original qualification is retained in Etymology.')
   actions.append(dict(entry_key=key,source_unit_key=u['source_unit_key'],component=c['named_component'],
                       disposition=c['disposition'],parent_keys=parents,reason=c['reason']))
# Bound morphology is an explicitly named source component, not a reconstructed
# stem. Only the occurrence naming it receives this edge.
for u in audit:
 if u['representation']!='explicit bound morphology':continue
 source=u['source_record']; head=audit_by_key[source['parent_key']]
 if source['parent_key']=='kholosi:pdf:p7:c1:u2':
  actions.append(dict(source_unit_key=u['source_unit_key'],disposition='productive suffix contextual only',
   reason='The petak etymology raises competing analyses; productive -ak is not confidently assigned to this particular head.'))
  continue
 for key in head['emitted_entry_keys']:
  row=bykey[key]
  for parent in u['emitted_entry_keys']:
   assert parent!=key
   row[13]=' '.join(dict.fromkeys(row[13].split()+[parent]))
  if 'uncertain' in source.get('tags',[]):
   uncertain(row)
   note(row,'The source marks this suffix analysis or its function as uncertain; the source analysis is preserved in the linked morphology witness.')
  actions.append(dict(entry_key=key,source_unit_key=u['source_unit_key'],
                       disposition='explicit bound component',parent_keys=u['emitted_entry_keys'],
                       source_analysis=source['source_analysis']))
for old,new in zip(baseline,rows):
 assert len(old)==len(new)==15
 assert all(old[i]==new[i] for i in (0,2,3,4,5,7,8,9,10,11,12))
 if old[10].startswith('legacy:'):assert old[1]==new[1]
with (R/'whole-graph-draft.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
(R/'row-graph-actions.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in actions))
(R/'draft-status.json').write_text(json.dumps(dict(status='not frozen; focused validation and independent review pending',
 rows=len(rows),new_ancestry_assignments=sum(a[1]!=b[1] for a,b in zip(baseline,rows)),
 changed_rows=sum(a!=b for a,b in zip(baseline,rows)),legacy_parameters_preserved=198,
 unchanged_fields=['Lang','Form','Gloss','Native','Phonemic','Source','Cognateset','Etymology','Entry_Key','Variant_Of_Key','Borrowed_From_Key'],
 draft_sha256=hashlib.sha256((R/'whole-graph-draft.csv').read_bytes()).hexdigest()),indent=2)+'\n')
print((R/'draft-status.json').read_text())
