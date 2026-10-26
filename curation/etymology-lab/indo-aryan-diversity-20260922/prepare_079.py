import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments

P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
records=json.loads((P/'grangali-remaining-current.json').read_text())
linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
imports=list(csv.reader(open('data/other/forms/20260819-buddruss-grangali.csv')))
source=list(csv.DictReader(open('data/other/forms/raw_data/20260819-buddruss-grangali-audit.csv')))
groups=[
('63','brother','9661','bhrātṛ','Turner9661 directly lists Ningalami lā and Shumashti lā brother. Buddruss records the Grangali la counterpart on the same item, and all three share the local l-vowel stem; source vowel lengths remain distinct.'),
('160','belly','1932','udara','Turner1932 directly lists Ningalami wōr and Shumashti war belly under udara. Buddruss records the identical Grangali vōr/Ningalami vōr pairing; v and w are retained as in their respective source transcription profiles.'),
('162','village','4368','grāma','Turner4368 directly lists Ningalami and Shumashti lām village, alongside Pashai lām and neighbouring Gawri lām. Buddruss records Grangali, Ningalami and Shumashti lăm; the source short-vowel notation is retained, without assuming a fresh borrowing.')]
accepted=[];audit=[]
lines=['# Grangali survey: brother, belly and village','','Nine reflex records across three families. Buddruss printed pp. 29 and 35 visually checked; complete Turner articles read.','','| # | Forms and meaning | Parent | Evidence |','|---|---|---|---|']
for i,(item,gloss,parent,label,evidence) in enumerate(groups,1):
 hits=[r for r in records if r['Gloss']==gloss and r['Source'].endswith(f'item {item}]')]
 assert len(hits)==3,(item,hits)
 for r in hits:
  assert r['ID'] not in linked,r
  key=f'buddruss-grangali1979:item-{item}'+{'Gng':'','Ning':':ningalami','Shum':':shumashti'}[r['Language_ID']]
  ir=[z for z in imports if z[0]==r['Language_ID'] and z[3]==gloss and z[10]==key]
  ar=[z for z in source if z['Emitted_Key']==key and z['Atlas_Numbers']==item and z['Material_Error']=='no']
  assert len(ir)==len(ar)==1,(r,ir,ar)
  assert ar[0]['Printed_Page']==r['Source'].split('p. ')[1].split(',')[0]
  accepted.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=f'buddruss-grangali1979[p. {ar[0]["Printed_Page"]}, item {item}];CDIAL[{parent}]',evidence=evidence+' Reviewed source audit and import row matched; no transcription or identity changes.'))
  audit.append(dict(record_id=r['ID'],source_audit=ar[0],import_row=ir[0],image=f'../tmp/pdfs/grangali/render/page-{int(ar[0]["Printed_Page"])-16:02d}.png'))
 lines.append(f'| {i} | **{" / ".join(r["Form"] for r in hits)}** ‘{gloss}’ (3) | **{label}**, CDIAL **{parent}** | {evidence} |')
assert len(accepted)==9
(P/'pass079-decisions.json').write_text(json.dumps({'accepted':accepted,'held':[]},ensure_ascii=False,indent=2)+'\n')
(P/'pass079-source-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
lines+=['','Discovery: grangali-search018; full articles grangali-primary016. Other source words in that query remain unassigned where a specific history was not established. No rebuild.','','Save and validation pending.']
(P/'pass079-review.md').write_text('\n'.join(lines)+'\n')
print(len(accepted),'accepted')
