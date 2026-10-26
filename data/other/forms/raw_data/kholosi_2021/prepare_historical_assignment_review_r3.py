"""Propose evidence-backed corrections; never mutate accepted historical rows."""
import csv
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'graph-reconciliation-r3'
R.mkdir(exist_ok=True)
old=list(csv.reader((P/'legacy-baseline/20220913-kholosi.csv').open()))
targets={r[0]:dict(head=r[1],definition=r[3]) for r in csv.reader((P.parents[4]/'data/cdial/params.csv').open())}
claims=list(map(json.loads,(P/'source-etymology-claims.jsonl').open()))
specs=[
 (92,'9735','Both source versions explicitly identify MAṆḌÁ9735. Old9375 identifies a tree, not the named head or bread meaning.'),
 (194,'3138','Printed source explicitly identifies KĀ́SĀ3138, cough. Old3865 means eat and belongs to a different verbal head.'),
 (141,'12357','PDF names ŚAŚÁ12357, hare; repository names śaśá but miscites12359, whose head is moon. The unique named head supports12357.'),
 (169,'13744','Both versions name STHÁLA but print13774. That number identifies a post/pillar; the unique named head sthála occurs at13744.'),
 (64,'','Both versions name KR̥TḖ but print2184. That number identifies upadhārayati, supports/holds up. No exact intended target is established, so remove the unsupported link while retaining the complete source claim.'),
]
records=[]
for n,new,reason in specs:
 row=old[n-1]
 assert row[10]==f'legacy:20220913-kholosi:row:{n}'
 source_form='moni' if n==92 else row[2]
 evidence=[c for c in claims if c['source_form']==source_form]
 assert evidence
 records.append(dict(entry_key=row[10],historical_row=row,old_parameter=row[1],proposed_parameter=new,
  old_target=targets[row[1]],proposed_target=targets[new] if new else None,
  source_claims=evidence,matched_current_source_form=source_form,reason=reason,status='proposed correction; independent review required',
  continuity='Entry key, literal form, IPA, gloss, citation and all other lexical fields remain unchanged. Historical assignment remains fully preserved in this audit.'))
report=dict(status='proposal only; frozen r2 and canonical unchanged',
 baseline_sha256=hashlib.sha256((P/'legacy-baseline/20220913-kholosi.csv').read_bytes()).hexdigest(),
 target_inventory_sha256=hashlib.sha256((P.parents[4]/'data/cdial/params.csv').read_bytes()).hexdigest(),
 corrections=records,
 retained_competing_analyses=[
  dict(row=63,form='kāti',old_id='3069',source_id='2853',reason='Both IDs designate knife ancestry; disagreement is not a demonstrably unrelated-head typo.'),
  dict(row=116,form='pāyo',old_id='7747',source_id='8056',reason='Repository explicitly names pada7747; PDF names pāda8056. Preserve distinct version claims.'),
  dict(row=147,form='sāl',old_id='4749',source_id='12415',reason='Repository and historical assignment support cāmala; PDF tentatively proposes śāli. No arbitrary winner.'),
 ])
(R/'historical-assignment-correction-proposal.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Proposed5 historical corrections with exact source and target evidence.')
