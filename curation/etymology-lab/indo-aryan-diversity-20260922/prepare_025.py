import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
rs=[{k:r[k] for k in ('ID','Language_ID','Form','Gloss','Source','Tags')} for r in csv.DictReader(open('cldf/forms.csv')) if 'buddruss-grangali1979[' in r['Source']]
assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('horse',['gōrə́','gōro'],'4516','ghoṭa','Turner explicitly gives Shumashti gṓṛo and Ningalam guṛə beside Pashai gōṛā. The source deaspirated forms continue this regional horse family; its proposed remote non-Aryan origin does not establish a recent donor for these forms.'),
('ear',['kõ','kõr'],'2830','karṇa','Turner explicitly gives Ningalam kõ and Shumashti kõṛ. Grangali agrees with the nasal-vowel Ningalam form; source r/ṛ notation is retained rather than silently regularised.'),
('barley',['zō','zo'],'10431','yava','Turner explicitly lists Ningalam zō and Shumashti zo under yava. Grangali zō matches this regional y-to-z outcome.'),
('blood',['lū','luí'],'11165','lohita','Turner explicitly gives Shumashti lúī and Pashai lū blood under lohita. The shorter lohi entry11164 instead lists Kalasha/Khowar comparisons; the same-language Shumashti witness supports11165 for this batch.'),
('tongue',['zíp','zip','zīb'],'5228','jihvā','Turner section1 explicitly lists Ningalam zip and Shumashti zīb beside Pashai zīp. The source bare tongue forms are preserved; parsed my-tongue gloss spillover does not justify adding a possessive component.'),
('twenty',['isí'],'11616','viṁśati','Turner explicitly gives Ningalam isī and Shumashti ísī twenty. The initial-v-loss regional series supports the three source records; recorded accent and length remain unchanged.'),
('three',['λē','sle'],'5994','trayaḥ','Turner section1 lists Ningalam ṣlē, Shumashti λyē and neighboring λē. The source Grangali λē and Ningalam/Shumashti sle belong to this lateral-cluster e-vowel series, not the trīṇi subsection5994-3. Differences in the recorded cluster notation remain explicit.'),
('blind',['andə́','andá'],'385','andha','Turner gives Shumashti anda and Pashai andá blind. Grangali andə agrees with this deaspirated regional adjective; the unrelated meat gloss andá is excluded.'),
('fire',['angā́r','angār'],'125','aṅgāra','Turner section1 gives Pashai and neighboring aṅgār fire under aṅgāra glowing charcoal. The search also identifies Ningalam aŋgār here. Grangali/Ningalam nasal-stop forms fit this family; the specifically loan-marked Shumashti ãr is excluded pending a supported immediate donor.'),
('sun',['surí'],'13574-3','sūriya','Turner section3 explicitly lists Ningalam surī and Shumashti súrī beside regional sūri. Use sūriya13574-3, not the sūra head or feminine sūrī subsection4.'),
('urine',['muλ'],'10234','mūtra','Turner explicitly gives Shumashti and Gawri muλ urine. The source Grangali/Shumashti lateral-fricative forms match the regional tr outcome without an added suffix.')]
a=[]
for gloss,fs,pa,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Gloss']==gloss and r['Form'] in fs:a.append(dict(record=r,parent=pa,kind='reflex',citation=f'CDIAL[{pa} and addenda];'+r['Source'],evidence=ev+' Existing full manual source-image census reports no material error; language label, accent and source transcription are preserved.'))
h=[]
for r in rs:
 if r['ID'] in assigned:continue
 if r['Gloss']=='bone':h.append(dict(record=r,reason='Turner958 explicitly treats the relevant Dardic forms, including Shumashti ãṭhi, as borrowed from Indic. Resolve the immediate donor before adding an ancestry link; do not skip to Sanskrit aṣṭhi.'))
 if r['Gloss']=='fire' and r['Language_ID']=='Shum':h.append(dict(record=r,reason='Turner125 explicitly labels Shumashti ãr as borrowed from Pashai. Resolve a supported existing immediate Pashai donor, rather than linking directly as a Sanskrit reflex.'))
assert len(a)==28,len(a)
assert len(h)==3,len(h)
(P/'pass025-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=2)+'\n')
lines=['# Grangali, Ningalam and Shumashti: further vocabulary','','28 records / 28 reflex rows across eleven etyma; three loan-route holds. Standing autonomous objective authorizes saving supported existing-node analyses.','','| # | Language forms and meanings | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,fs,pa,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==pa for x in a);lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({n}) | **{label}**, CDIAL **{pa}** | {ev} |')
lines+=['','Source evidence: existing 20260819-buddruss-grangali-audit.csv documents a full manual census against300dpi page images. All selected readings have Material_Error=no; corresponding audit rows are retained in pass025-source-audit.json. Shared source locators can also contain a separate language word; exact form/gloss matching excludes it.','', 'Holds: two bone records need immediate Indic donor resolution (Turner958 explicitly marks this regional group as loans); Shumashti fire ãr needs a Pashai donor (explicitly marked in125). Earlier horn holds remain unchanged. No loan is flattened to a Sanskrit reflex.','', 'Research: grangali-search002–004.json; grangali-primary002–004.json, including full primary prose and addenda. No source ingestion, notation repair, identity changes or rebuild.','', 'Validation and save status: pending.']
(P/'pass025-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(h),'held')
