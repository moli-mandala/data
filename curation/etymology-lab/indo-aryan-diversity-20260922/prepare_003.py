import json,csv
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
records=json.loads((P/'grangali-records.json').read_text())
assigned={r['Form_ID'] for r in read_assignments()}
assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('smoke',['dum','dūm'],'6849','dhūma','Turner explicitly gives Shumashti dūm and Ningalam dum under section 1 dhūma. Deaspirated d and retained m identify this family; these are not the feminine dhūmikā forms in section 2.'),
('hand',['ast','vōst'],'14024','hasta','Turner gives Shumashti aste-m and Ningalam wōst, explaining the latter initial w as extension from wōr belly to body-part names. Grangali/Shumashti ast retains st with loss of h; the bare source stems require no possessive component.'),
('moon; month',['mas'],'10104','māsa','Turner explicitly lists Ningalam mas moon, beside regional mas moon/month. Retention of s matches this Dardic series; the distinct expanded Grangali masulī is excluded.'),
('snow',['ǐm','im'],'14096','hima','Turner explicitly lists Ningalam im and Shumashti īm snow, alongside Pashai hīm/īm/yīm. Initial h loss supports the Grangali form as well. Buddruss prints Grangali i with a caron; its notation is preserved without inventing an extra morpheme.'),
('four',['ʦōr','ʦᵘor'],'4655-3','caturaḥ','Turner section 3 gives Ningalam ċᵘōr and neighboring ċōr/ċōur, continuing the caura series. The rounded vowel distinguishes this subsection from the cār series under catvāri.'),
('eight',['aṣṭ','õṣṭ','ãṣṭ'],'941','aṣṭā','Turner gives Shumashti ãṣṭ and Dardic aṣṭ/oṣṭ comparanda. The retained ṣṭ cluster identifies the inherited numeral family; the source nasal vowels remain as printed.'),
('salt',['lõ','lon'],'10978','lavaṇa','Turner explicitly gives Ningalam lō̃ and Shumashti lōn; compare Pk lōṇa. Grangali lõ agrees with the neighboring nasal-vowel outcome after loss of the final nasal consonant.'),
('fish',['maʦ','mōʦ','māʦ'],'9758','matsya','Turner section 1 explicitly gives Ningalam mōċ and Shumashti māċ beside Pashai māċ. Grangali maʦ matches the unextended affricate stem; no -i or -l extension is selected.')]
a=[]
for gloss,forms,parent,label,ev in groups:
 for r in records:
  if r['Form'] in forms and r['Gloss']==gloss and r['ID'] not in assigned:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev))
held=[dict(record=r,reason='Candidate CDIAL 12583 śṛṅga; compare Turner Shumashti ṣīṅ. Buddruss gives ṣín with final n. Check the nasal-place difference and local sound history before saving.') for r in records if r['Gloss']=='horn' and r['ID'] not in assigned]
(P/'pass003-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines=['# Grangali, Ningalam and Shumashti: first diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; eight etyma. No rebuild.','','| # | Language forms and meanings | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(gloss,forms,parent,label,ev) in enumerate(groups,1):
 rs=[x['record'] for x in a if x['parent']==parent];display=', '.join(r['Language_ID']+' **'+r['Form']+'**' for r in rs)
 lines.append(f'| {i} | {display} ‘{gloss}’ | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['',f'{len(held)} horn records held over the final nasal correspondence. No loans or circumlocutions included. The source is the existing manually collated Buddruss transcription; no source re-ingestion is performed.','', 'Validation and save status: pending.']
(P/'pass003-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
