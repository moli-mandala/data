import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'pashai-records.json').read_text())
assigned={r['Form_ID'] for r in read_assignments()};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['māc'],['fish'],'9758','matsya','Turner section 1 gives Pashai ar. māċ and dar. māč fish, beside Ningalam mōċ and Shumashti māċ. The bare affricate stem matches this section rather than *matsiya or an extended form.'),
(['dum','dū̌m'],['smoke'],'6849','dhūma','Turner section 1 explicitly gives Pashai gul. dūm and Shumashti dūm, with loss of aspiration but retention of m. Source vowel and accent spellings are preserved as separate attestations.'),
(['im','ī̌m'],['snow'],'14096','hima','Turner explicitly gives Pashai hīm, īm, yīm snow; the h-less source forms fit the īm series. Accent/length notation is retained unchanged.'),
(['lōṇ','loṇ'],['salt'],'10978','lavaṇa','Turner explicitly lists Pashai dar. lōṇ salt beside other Pashai lūn/lεn variants. The contracted vowel and retained retroflex nasal identify this family.'),
(['gōṛā','gōṛi'],['horse','mare'],'4516','ghoṭa','Turner gives Pashai gōṛā horse and the corresponding masculine/feminine horse series across the region. The source mare has ordinary feminine i; no derivational or loan route is invented.'),
(['dān'],['tooth'],'6152','danta','Turner gives Pashai dan(d) tooth, with neighboring loss or retention of the final dental. The source dān agrees with the simplified nasal-final form.'),
(['gōm'],['wheat'],'4287','godhūma','Turner gives Pashai gūm and Ningalam/Shumashti gōom wheat, with many neighboring gōm forms. Loss of medial dh and contraction account for the shape.'),
(['may','māi'],['month','moon'],'10104','māsa','Turner explicitly gives Pashai lauṛ. māya/mā̆i and dar. māi moon/month. The source preserves the two related senses as distinct records.'),
(['panj'],['five'],'7655','pañca','Turner gives Pashai pā́nǰa and pōnǰ five. The source nasal plus voiced affricate belongs to this inherited numeral family.'),
(['sat'],['seven'],'13139','sapta','Turner explicitly gives Pashai dar. and chil. sat seven, with pt > tt/t and loss of the final vowel.'),
(['do'],['two'],'6648','dva','Turner explicitly lists Pashai dō/dö/dū two; source do belongs to the monophthongal outcome of the duv-/dv- series.')]
a=[]
for forms,glosses,parent,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in forms and r['Gloss'] in glosses:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];lehr',evidence=ev+' Lehr records are the installed Darrai-i Nur attestations; the existing source citation has no page locator, and no locator is invented.'))
held=[]
for r in rs:
 if r['ID'] in assigned:continue
 if r['Form'] in ['ṣēr','śir'] and r['Gloss']=='milk':held.append(dict(record=r,reason='CDIAL 3696 kṣīra includes Pashai ṣīr, but the source forms need comparison with Iranian milk loans before the immediate relation is fixed.'))
 if r['Form']=='nām' and r['Gloss']=='name':held.append(dict(record=r,reason='CDIAL 7067 nāman supplies Pashai nām, but Persian nām is also possible in this contact setting; retain pending source-specific evidence.'))
 if r['Form'] in ['dār','dā̌r'] and r['Gloss']=='mountain':held.append(dict(record=r,reason='Search candidate darad 6191 only gives Marathi daraḍ and does not establish the Pashai derivation; do not assign on resemblance alone.'))
(P/'pass008-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines=['# Pashai (Darrai-i Nur): first diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; eleven etyma. No rebuild.','','| # | Pashai | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(forms,glosses,parent,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==parent for x in a);lines.append(f'| {i} | **{" / ".join(forms)}** ‘{" / ".join(glosses)}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['',f'{len(held)} records held: milk, name and mountain need stronger discrimination. Existing source citations have no page locators; none are invented. No multiword responses or mixed-gloss copula/hand records are included.','', 'Validation and save status: pending.']
(P/'pass008-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
