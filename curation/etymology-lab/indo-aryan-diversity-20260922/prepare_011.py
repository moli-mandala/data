import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'kisan-records.json').read_text());assigned={r['Form_ID'] for r in read_assignments()};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['aĩkh','aĩkʰ'],['eye'],'43','akṣi','Turner gives eastern ā̃khi/ākhi and Oriya āẽkha eye, supporting the nasal-vowel and kh family. Source aĩ and its two aspiration spellings are preserved; no vowel sequence is silently respelled.'),
(['kan'],['ear'],'2830','karṇa','Turner gives regional kān ear via MIA kaṇṇa. The source kan has the inherited k-n frame and exact ear meaning; its unmarked vowel length remains unaltered.'),
(['nak'],['nose'],'6909','*nakka','Turner gives Nepali/Maithili/Bhojpuri nāk nose and MIA ṇakka. The source final k identifies this established family; no length mark is added.'),
(['kes','kẽs'],['hair'],'3471','keśa','Turner gives Nepali/Maithili kes and Marathi kes/kẽs hair beside MIA kesa. Both source nasalisation variants fit the hair family; tsul is not assigned here.'),
(['dzib','dzibha'],['tongue'],'5228','jihvā','Turner gives Bengali jib, Oriya jibha and Bhojpuri jībhⁱ, from MIA jibbhā. The source dz spelling of the initial affricate and its final b/bha variation are retained.'),
(['pet'],['belly'],'8376','*peṭṭa','Turner section 1 gives Nepali and eastern peṭ belly beside MIA peṭṭa. The source plain t is preserved; the e vowel distinguishes this from the poṭṭa section.'),
(['gor'],['leg'],'4272','*goḍḍa','Turner gives Maithili goṛ and Bhojpuri gōṛ foot/leg beside Nepali goṛo. The source r notation is retained without inserting a retroflex diacritic.'),
(['gʰar','ghar'],['house'],'4428','ghara','Turner explicitly gives Nepali/Bihari ghar and Maithili ghar/gharo house. The two source aspiration spellings remain distinct attestations, and the parent is ghara rather than a direct leap to gṛha.'),
(['kath̪i','kathi'],['firewood'],'3120','kāṣṭha','Turner includes kāṣṭhikā and regional kāṭhī wood. The source firewood sense matches that family; its dental/plain t notation is preserved rather than normalised to an inferred retroflex.'),
(['dãt'],['teeth'],'6152','danta','Turner gives Nepali/Maithili/Bhojpuri dā̃t tooth, from the danta family. Only the intact source dãt reading is included; records with detached dental marks remain unmodified and unassigned.')]
a=[]
for forms,glosses,parent,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in forms and r['Gloss'] in glosses:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev+' The source transcription/gloss uncertainty tag remains; this assignment does not claim a corrected phonemic transcription.'))
aud={}
for line in open('data/other/forms/raw_data/more_surveys_2026/kisan-audit.jsonl'):
 row=json.loads(line)
 for reading in row.get('readings',[]):
  ins=reading.get('row',[])
  if len(ins)>7:aud.setdefault(ins[7],[]).extend(reading.get('review',[]))
for x in a:
 notes=aud[x['record']['Source']]
 assert notes and all(n=='transcription/gloss: source attestation preserved; inconsistent notation or prompt meaning is not silently emended' for n in notes),(x['record'],notes)
held=[dict(record=r,reason='Detached phonetic marks/spaces in the source tooth record require a reading audit before etymological assignment; no transcription repair is made here.') for r in rs if r['Gloss']=='teeth' and r['Form']!='dãt' and r['ID'] not in assigned]
(P/'pass011-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n');(P/'pass011-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(a),uncertaintyRetained=True,sources={x['record']['Source']:aud[x['record']['Source']] for x in a}),ensure_ascii=False,indent=2)+'\n')
lines=['# Kisan Indo-Aryan survey: first diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; ten etyma. No rebuild.','','| # | Kisan Indo-Aryan | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(forms,glosses,parent,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==parent for x in a);label=label.replace('*',r'\*');lines.append(f'| {i} | **{" / ".join(forms)}** ‘{" / ".join(glosses)}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['',f'{len(held)} tooth records held over detached notation. Source uncertainty tags, dialect labels, glosses and spellings remain unchanged. These are the KisanIA survey records, not the Dravidian Kisan language.','', 'Validation and save status: pending.']
(P/'pass011-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
