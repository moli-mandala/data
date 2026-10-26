import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'majhi-records.json').read_text());assigned={r['Form_ID'] for r in read_assignments()};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['ā̃kʰi'],['eye'],'43','akṣi','Turner gives Bihari/Maithili ā̃khi and Nepali ā̃khi/ā̃kho. The nasal vowel, aspirated velar and final i identify the akṣi > akkhi family.'),
(['kān'],['ear'],'2830','karṇa','Turner gives Nepali, Maithili and Bhojpuri kān ear via MIA kaṇṇa. The unextended source form fits this family; kānek is not silently stripped of its ending.'),
(['nāk'],['nose'],'6909','*nakka','Turner explicitly gives Nepali/Maithili nāk nose under *nakka, with final k retained after historical gemination. Source nākno/nāknok require separate morphological treatment.'),
(['d̪ā̃t̪'],['teeth'],'6152','danta','Turner gives Nepali/Maithili dā̃t tooth. The source dental diacritics and plural elicitation gloss are preserved; the -ek response is not included.'),
(['hāt̪'],['arm/ hand'],'14024','hasta','Turner gives Nepali hāt hand/forearm beside Bihari hāth. Source dental t and the full arm/hand gloss match that family, without adding an unsupported component.'),
(['gʰar'],['house'],'4428','ghara','Turner explicitly gives Nepali, Bihari and Maithili ghar/gharo house. The parent is ghara; its debated deeper relationship to gṛha is not bypassed.'),
(['mut̪'],['urine'],'10234','mūtra','Turner gives Nepali mut and Bihari mūt urine beside MIA mutta, supporting simplification of tr to a dental stop. The source dental notation is retained.'),
(['gu'],['feces'],'4225','gūtha','Turner section 1 gives Nepali gū/guhu and regional gū excrement, with loss of medial th through h. This is distinct from the *gūttha section retaining t.'),
(['kātʰi'],['firewood'],'3120','kāṣṭha','Turner covers kāṣṭhikā and Sindhi/Hindi kāṭhī wood in this entry. The source firewood word fits this wood/stick family; its printed plain t is retained rather than silently converted to a retroflex.'),
(['pāni'],['water','rain'],'8082','pānīya','Turner gives Nepali/Bihari pāni water, and the addendum explicitly includes water/rain. Both source senses remain separate records, with no gloss replacement.'),
(['bār'],['hair'],'11572','vāla','Turner explicitly gives Bhojpuri bār hair alongside Nepali bāl, supporting the regional l/r outcome. The other hair responses are not pooled under this analysis.'),
(['pet'],['belly'],'8376','*peṭṭa','Turner section 1 gives Nepali peṭ belly and MIA peṭṭa. The source plain t is preserved; this family is distinct from poṭṭa and the lower-belly *peḍḍuka section.')]
a=[]
for forms,glosses,parent,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in forms and r['Gloss'] in glosses:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev+' The installed source uncertainty concerns its published phonetic notation; no guessed phonemic value or transcription repair is introduced.'))
aud={}
for line in open('data/other/forms/raw_data/selected_surveys_2026/majhi-audit.jsonl'):
 row=json.loads(line)
 for reading in row.get('readings',[]):
  ins=reading.get('row',[])
  if len(ins)>7:aud.setdefault(ins[7],[]).extend(reading.get('review',[]))
for x in a:
 notes=aud[x['record']['Source']]
 assert notes and all(n=='transcription: published notation retained; ambiguous symbols are not assigned guessed phonemic values' for n in notes),(x['record'],notes)
held=[dict(record=r,reason='Additional -ek/-no/-nok material requires source morphology before the whole response can be linked.') for r in rs if r['Form'] in ['kānek','nāknok','nākno','d̪ā̃t̪ek','hāt̪ek'] and r['ID'] not in assigned]
(P/'pass010-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
(P/'pass010-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(a),retainedPublishedNotation=True,sources={x['record']['Source']:aud[x['record']['Source']] for x in a}),ensure_ascii=False,indent=2)+'\n')
lines=['# Majhi survey: first diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; twelve etyma. No rebuild.','','| # | Majhi | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(forms,glosses,parent,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==parent for x in a);label=label.replace('*',r'\*');lines.append(f'| {i} | **{" / ".join(forms)}** ‘{" / ".join(glosses)}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['',f'{len(held)} extended responses held for morphology. All selected source-audit notes were checked; published notation and uncertainty tags are retained. Individual locality attestations remain separate.','', 'Validation and save status: pending.']
(P/'pass010-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
