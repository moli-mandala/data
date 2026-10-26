import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'zargari-records.json').read_text())
assigned={r['Form_ID'] for r in read_assignments()};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['pani'],['water'],'8082','pānīya','The source explicitly compares Sanskrit pānīya and Hindi pānī; Turner lists European Romani pāni water. The Hindi form is a comparandum, not a claimed immediate donor.'),
(['kālo'],['black; dark'],'3083','kāla','Turner explicitly gives European Romani kalo black/dark beside Pk kāla. The source itself compares the Romani and Hindi forms. Turner discusses a possible deeper Dravidian origin, which this inherited family link does not resolve.'),
(['per'],['belly; stomach'],'8376','*peṭṭa','Turner specifically gives Greek/Rumanian/Hungarian Romani per belly under section 1 *peṭṭa, distinct from poṭṭa navel/belly in section 3. The source compares peṭ belly and peṭa basket; Turner leaves the deeper relation with the basket words open.'),
(['yākʰ'],['eye'],'43','akṣi','Turner gives European Romani yakh eye, explaining y from the preceding feminine article or adjective. The inherited Romani stem is linked as a lexical unit, not split into a modern article component.'),
(['vāst'],['arm; hand'],'14024','hasta','The source explicitly derives this from Old Indo-Aryan hasta. Turner gives European Romani vast hand and explains v from earlier ov ast; the source arm/hand polysemy is preserved.'),
(['kʰer'],['house'],'4428','ghara','Turner explicitly gives European Romani kher house. The graph parent is ghara; the historically difficult deeper connection to gṛha is not skipped.'),
(['māccʰo'],['fish'],'9758','matsya','The source compares Early Romani mačho. Turner section 1 includes southeastern European Romani mačhó alongside mačo; the affricate and final o fit this family, not the *matsiya subsection.'),
(['kāśt'],['wood','tree'],'3120','kāṣṭha','Turner gives European Romani kašt piece of wood/timber. The source explicitly compares Sanskrit kāṣṭha and a Qorbati tree/wood word for the tree sense. The wood/tree source records remain distinct attestations.'),
(['cʰib'],['language; tongue'],'5228','jihvā','Turner section 1 gives southeastern European Romani čhib tongue. The source language/tongue polysemy is retained; this is not section 2 juhū.'),
(['lon'],['salt'],'10978','lavaṇa','Turner explicitly gives European Romani lon salt beside Middle Indo-Aryan lōṇa, with medial v loss and contraction.'),
(['yiv'],['snow'],'14096','hima','Turner gives European Romani hiv, iv, yiv snow, directly matching the Zargari inherited family.'),
(['tʰuv'],['smoke'],'6849','dhūma','Turner section 1 explicitly gives European Romani thuv smoke, with the Romani devoiced aspirate and final v corresponding to dh and m.'),
(['mās'],['meat'],'9982','māṃsa','The source explicitly cites Sanskrit māṃsa and Vedic mās. Turner gives Pk māsa and European Romani mas flesh; nasal-cluster simplification and loss of the final vowel fit.'),
(['ruv'],['wolf'],'10754','*ruka','Turner has a dedicated *ruka wolf entry for European Romani ruv, compared with vṛka. This is the specific existing etymon; the homonymous ruv cry imperative is excluded.'),
(['dānd'],['tooth'],'6152','danta','Turner explicitly gives European Romani dand tooth. The retained nasal plus voiced dental stop fits this inherited series.')]
a=[]
for forms,glosses,parent,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in forms and r['Gloss'] in glosses:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev))
base=next(r for r in rs if r['Form']=='cʰib' and r['Gloss']=='language; tongue')
for r in rs:
 if r['ID'] not in assigned and r['Form']=='cʰiv' and r['Gloss']=='language; tongue':a.append(dict(record=r,parent=base['ID'],kind='variant',citation=r['Source'],evidence='The source explicitly derives čhiv from čhib and lists further čhip/čiph developments. Link this source variant to the existing čhib language/tongue record, whose jihvā etymology is saved in this pass.'))
held=[dict(record=r,reason='CDIAL 4501 gives European Romani khil fat/butter. The source gloss is oil; verify the intended substance and semantic extension before linking.') for r in rs if r['Form']=='kʰil' and r['ID'] not in assigned]
(P/'pass006-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines=['# Zargari: first diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and rows; fifteen etyma and one source-explicit variant link. No rebuild.','','| # | Zargari | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(forms,glosses,parent,label,ev) in enumerate(groups,1):
 label=label.replace('*',r'\*');n=sum(x['parent']==parent for x in a)
 lines.append(f'| {i} | **{" / ".join(forms)}** ‘{" / ".join(glosses)}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['| 16 | **cʰiv** ‘language; tongue’ | Variant of source **cʰib** | The source explicitly gives čhiv < čhib. The variant points to that attestation, with its ancestry supplied by row 9. |','','kʰil oil is held over semantic scope. Loanword records, compounds and homonymous cry ruv are excluded.','', 'Validation and save status: pending.']
(P/'pass006-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
