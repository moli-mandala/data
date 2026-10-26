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
('114','stone',{'Gng':'11348','Ning':'11348'},'*varta³','Turner11348 section1 directly lists Ningalami bōt stone. Buddruss records identical Grangali and Ningalami bōt and explicitly contrasts this stone word with the cooked-rice homophone.'),
('137','red',{'Gng':'1296','Ning':'2209','Shum':'1296'},'ārakta / uparakta','Turner1296 directly gives Shumashti aratə́ red under ārakta; Buddruss identifies Grangali artə́ with the same arakta family. For Ningalami vartə́, Turner2209 specifically gives uparakta. Buddruss groups the three source forms under arakta generally; the prefixed Ningalami assignment follows Turner’s more specific analysis.'),
('145','sister',{'Gng':'13913','Ning':'13913'},'svasṛ','Turner13913 directly lists Ningalami pas sister under svasṛ, alongside neighbouring Dardic pas forms. Buddruss records the identical Grangali and Ningalami pas pair.'),
('115','rain',{'Gng':'11392','Shum':'11392'},'varṣa','Turner11392 directly gives Shumashti wāṣ rain and other Dardic b/waṣ forms. Buddruss Grangali boṣ and Shumashti vas fit this rain family. Turner cautions that some Dardic long-vowel forms could instead continue vārṣa; that historical alternative is retained in the note.'),
('138','sand',{'Gng':'13386','Shum':'13386'},'sikatā','Turner13386 directly lists Shumashti sī́u sand alongside Pashai sēo/sīu. Buddruss Grangali seóu belongs to this regional formation. Turner warns that Iranian influence can affect the s-initial Dardic forms; this assignment does not settle the deeper contact history.')]
accepted=[];audit=[]
lines=['# Grangali survey: stone, red, sister, rain, sand and louse','','Eleven reflex records across five families plus one explicit Pashai-to-Shumashti loan. Buddruss printed pp. 32–34 visually checked; full Turner articles read.','','| # | Forms and meaning | Parent | Evidence |','|---|---|---|']
for i,(item,gloss,parents,label,evidence) in enumerate(groups,1):
 hits=[r for r in records if r['Gloss']==gloss and r['Source'].endswith(f'item {item}]')]
 assert len(hits)==len(parents),(item,hits)
 for r in hits:
  assert r['ID'] not in linked,r
  key=f'buddruss-grangali1979:item-{item}'+{'Gng':'','Ning':':ningalami','Shum':':shumashti'}[r['Language_ID']]
  ir=[z for z in imports if z[0]==r['Language_ID'] and z[3]==gloss and z[10]==key]
  ar=[z for z in source if z['Emitted_Key']==key and z['Atlas_Numbers']==item and z['Material_Error']=='no']
  assert len(ir)==len(ar)==1,(r,ir,ar)
  assert ar[0]['Printed_Page']==r['Source'].split('p. ')[1].split(',')[0]
  parent=parents[r['Language_ID']]
  accepted.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=f'buddruss-grangali1979[p. {ar[0]["Printed_Page"]}, item {item}];CDIAL[{parent}]',evidence=evidence+' Reviewed source audit and import row matched; no transcription or identity changes.'))
  audit.append(dict(record_id=r['ID'],source_audit=ar[0],import_row=ir[0],image=f'../tmp/pdfs/grangali/render/page-{int(ar[0]["Printed_Page"])-16:02d}.png'))
 lines.append(f'| {i} | **{" / ".join(r["Form"] for r in hits)}** ‘{gloss}’ ({len(hits)}) | **{label}**, CDIAL **{", ".join(sorted(set(parents.values())))}** | {evidence} |')
donor='f_w7smlmkpdmi5a'
dr=[r for r in csv.DictReader(open('cldf/edges.csv')) if r['Child_ID']==donor and r['Parent_ID']=='10512' and r['Rank']=='1']
assert len(dr)==1
louse=[r for r in records if r['Language_ID']=='Shum' and r['Gloss']=='louse' and r['Source'].endswith('item 118]')]
assert len(louse)==1 and louse[0]['ID'] not in linked
r=louse[0];key='buddruss-grangali1979:item-118:shumashti'
ir=[z for z in imports if z[0]=='Shum' and z[3]=='louse' and z[10]==key]
ar=[z for z in source if z['Emitted_Key']==key and z['Atlas_Numbers']=='118' and z['Material_Error']=='no']
assert len(ir)==len(ar)==1
accepted.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=donor,kind='borrowed',citation='buddruss-grangali1979[p. 32, item 118];CDIAL[10512; NTS xiii 243]',evidence='Turner10512 explicitly labels Shumashti yū a loan from Pashai Darrai yū. The existing Pashai yū node has a rank-1 reflex from 10512 and is the immediate attested donor; Buddruss gives the Shumashti yū comparison. Reviewed source audit and import row matched; no donor node or transcription change.'))
audit.append(dict(record_id=r['ID'],source_audit=ar[0],import_row=ir[0],image='../tmp/pdfs/grangali/render/page-16.png',donor_id=donor,donor_edge=dr[0]))
lines.append(f'| 6 | **{r["Form"]}** ‘louse’ (1) | Borrowed from Pashai Darrai **{donor}** | Turner10512 explicitly labels Shumashti yū a loan from Pashai Darrai yū; the existing donor has rank-1 ancestry under 10512. Grangali zū remains unassigned. |')
assert len(accepted)==12
(P/'pass069-decisions.json').write_text(json.dumps({'accepted':accepted,'held':[]},ensure_ascii=False,indent=2)+'\n')
(P/'pass069-source-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
lines+=['','Discovery: grangali-search016; full primary articles grangali-primary014. Grangali zū louse remains unresolved; Shumashti yū is linked to the cited immediate Pashai donor. No rebuild.','','Save and validation pending.']
(P/'pass069-review.md').write_text('\n'.join(lines)+'\n')
print(len(accepted),'accepted')
