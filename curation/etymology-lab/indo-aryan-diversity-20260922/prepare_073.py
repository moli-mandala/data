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
('7','go','4939','cyavate','Turner4939 directly gives Shumashti ċō di go and neighboring Dardic čō go/imperative. Buddruss pairs Shumashti co with Grangali ca-, supplying a Grangali present and preterite to show it is a verbal stem; these forms belong to the local c- go family, without assigning the whole inflectional paradigm to a single ancestral tense.'),
('20','beat','11004','lāgayati','Turner11004 directly gives Shumashti lā̆y- beat/strike and neighbouring Ashkun lā- beat. Buddruss pairs Shumashti lay- with Grangali la- and gives Grangali present and preterite forms. The source stem spellings are preserved without claiming a precise tense derivation.')]
accepted=[];audit=[]
lines=['# Grangali survey: go and beat verbs','','Four reflex records across two verb families. Buddruss printed pp. 25–26 visually checked; complete Turner articles read.','','| # | Forms and meaning | Parent | Evidence |','|---|---|---|---|']
for i,(item,gloss,parent,label,evidence) in enumerate(groups,1):
 hits=[r for r in records if r['Gloss']==gloss and r['Source'].endswith(f'item {item}]')]
 assert len(hits)==2,(item,hits)
 for r in hits:
  assert r['ID'] not in linked,r
  key=f'buddruss-grangali1979:item-{item}'+{'Gng':'','Shum':':shumashti'}[r['Language_ID']]
  ir=[z for z in imports if z[0]==r['Language_ID'] and z[3]==gloss and z[10]==key]
  ar=[z for z in source if z['Emitted_Key']==key and z['Atlas_Numbers']==item and z['Material_Error']=='no']
  assert len(ir)==len(ar)==1,(r,ir,ar)
  assert ar[0]['Printed_Page']==r['Source'].split('p. ')[1].split(',')[0]
  accepted.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=f'buddruss-grangali1979[p. {ar[0]["Printed_Page"]}, item {item}];CDIAL[{parent}]',evidence=evidence+' Reviewed source audit and import row matched; no transcription or identity changes.'))
  audit.append(dict(record_id=r['ID'],source_audit=ar[0],import_row=ir[0],image=f'../tmp/pdfs/grangali/render/page-{int(ar[0]["Printed_Page"])-16:02d}.png'))
 lines.append(f'| {i} | **{" / ".join(r["Form"] for r in hits)}** ‘{gloss}’ (2) | **{label}**, CDIAL **{parent}** | {evidence} |')
assert len(accepted)==4
(P/'pass073-decisions.json').write_text(json.dumps({'accepted':accepted,'held':[]},ensure_ascii=False,indent=2)+'\n')
(P/'pass073-source-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
lines+=['','Discovery: grangali-search017; full articles grangali-primary015. Other verbs from the same query remain unassigned because no sufficiently specific historical comparison was found. No rebuild.','','Save and validation pending.']
(P/'pass073-review.md').write_text('\n'.join(lines)+'\n')
print(len(accepted),'accepted')
