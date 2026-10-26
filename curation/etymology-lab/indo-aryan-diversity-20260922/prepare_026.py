import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
rs=[{k:r[k] for k in ('ID','Language_ID','Form','Gloss','Source','Tags')} for r in csv.DictReader(open('cldf/forms.csv')) if 'buddruss-grangali1979[' in r['Source']]
assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('year',['kal','kāl'],'3084','kāla','reflex','Turner explicitly gives Shumashti kāl year under kāla, alongside Pashai kāl/kōl and regional year words. Grangali kal agrees with this regional sense; preserve its unmarked length. General Sanskrit-contact influence noted by Turner does not identify a separate immediate donor for these two records.'),
('winter',['ēmand'],'14164','hemanta','reflex','Turner section1 explicitly gives Ningalam ēmand winter. The source final nd and ē vowel select the attested hemanta branch rather than the proposed lengthened *hemānta subsection.'),
('egg',['ãr̃ə','vā̃na','ā̃ra'],'1111','āṇḍa','reflex','Turner explicitly gives Ningalam wāṇa and Shumashti nasal āṛa under āṇḍa; Grangali’s nasal r form agrees with the latter series. The heading discusses an *āṇḍra preform for regional forms, retained as historical uncertainty within this existing entry. Specific loan labels on other Pashai/Nuristani forms are not extended to these attestations.'),
('spring (season)',['basə́n','bəsə́n'],'11439','vasanta','reflex','Turner gives Dameli basán, Wotapuri basán and neighboring Gawri wasan spring under vasanta. Grangali basən fits this regional b-s-n seasonal family; preserve the source schwa and accent.'),
('cooked rice',['băt','bōt'],'9331','bhakta','reflex','Turner explicitly gives Ningalam bōt and Shumashti bät cooked rice, alongside Gawri bat and Shina băt. The Grangali form matches this food family. The distinct stone homonym bōt is excluded.'),
('mouse',['mūsõ'],'10258','mūṣa','reflex','Turner section1 explicitly gives Shumashti múṣo and neighboring mus-/muṣ- mouse forms. The source plain s and nasal final vowel are retained in this regional family assignment; no -ika suffix motivates10258-2.'),
('fire',['ãr'],'f_hgcxxts4qswly','Pashai aŋgā́r','borrowed','Turner125 explicitly marks Shumashti nasal ār as borrowed from Pashai. The same article’s Pashai aṅgār fire is the existing donor node f_hgcxxts4qswly (canonical Pas, no narrower dialect asserted), verified with rank1 reflex ancestry125. Link to this immediate donor rather than directly to Sanskrit aṅgāra; no donor node is created.')]
a=[]
for g,fs,pa,label,kind,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Gloss']==g and r['Form'] in fs:
   cite='125' if kind=='borrowed' else pa
   a.append(dict(record=r,parent=pa,kind=kind,citation=f'CDIAL[{cite} and addenda];'+r['Source'],evidence=ev+' Existing manually collated source record and lect label preserved.'))
h=[]
for r in rs:
 if r['ID'] not in assigned and r['Gloss']=='winter' and r['Form'] in ['imā́n','yeman']:
  reason='Grangali imān could reflect hemanta or the proposed lengthened *hemānta; resolve the subsection and possible Pashai route rather than copying the Torwali resemblance.' if r['Language_ID']=='Gng' else 'Turner explicitly says Shumashti yeman is borrowed from Pashai Chilas or Kurangal. Resolve an existing donor matching that attribution; do not choose another dialect solely by spelling.'
  h.append(dict(record=r,reason=reason))
(P/'pass026-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=2)+'\n')
sources={x['record']['Source'] for x in a+h};aud=[r for r in csv.DictReader(open('data/other/forms/raw_data/20260819-buddruss-grangali-audit.csv')) if r['Source'] in sources];assert all(r['Material_Error']=='no' for r in aud)
(P/'pass026-source-audit.json').write_text(json.dumps(aud,ensure_ascii=False,indent=2)+'\n')
lines=['# Grangali survey: seasonal vocabulary and a resolved loan','',f'{len(a)} records / {len(a)} rows: {sum(x["kind"]=="reflex" for x in a)} reflex and one borrowed link. Two winter records held. Existing compiled/accepted links excluded.','','| # | Forms and meanings | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,fs,pa,label,kind,ev) in enumerate(groups,1):
 n=sum(x['parent']==pa for x in a);proposal=f'Borrowed from **{label}**, existing node **{pa}**' if kind=='borrowed' else f'**{label}**, CDIAL **{pa}**';lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({n}) | {proposal} | {ev} |')
lines+=['','Source audit: existing full manual image census; all matching selected audit records have Material_Error=no. The Shumashti fire hold from pass025 is resolved by the existing Pashai entry. Bone and horn holds remain. Winter holds distinguish uncertain stem selection from explicit dialect-specific borrowing.','', 'Evidence files: grangali-search005.json; grangali-primary005.json plus full125 in grangali-primary004.json; pass026-source-audit.json. No new lexical nodes, source repairs or rebuild.','', 'Validation and save status: pending.'];(P/'pass026-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(h),'held')
