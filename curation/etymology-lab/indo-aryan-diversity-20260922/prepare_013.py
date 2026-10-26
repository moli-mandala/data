import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'angika-records.json').read_text());assigned={r['Form_ID'] for r in read_assignments()};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['ãkʰa','āĩkʰa'],['eye'],'43','akṣi','Turner gives eastern ā̃khi/ākhi and Oriya āẽkha eye. The nasal/diphthongal vowels plus kh identify this family; final a and the source vowel sequences are preserved.'),
(['kān'],['ear'],'2830','karṇa','Turner explicitly gives Maithili and Bhojpuri kān ear, from the karṇa > kaṇṇa series with cluster simplification.'),
(['nāk'],['nose'],'6909','*nakka','Turner gives Maithili/Bhojpuri nāk nose under *nakka. The source long vowel and final k match that series.'),
(['kes'],['hair'],'3471','keśa','Turner gives Maithili and Nepali kes hair beside Pali/Prakrit kesa, supporting s for ś and loss of the final vowel.'),
(['mu','muha','muhu'],['face','mouth'],'10158','mukha','Turner gives MIA muha and eastern mu/muh/mũh mouth/face. These source forms show the regular loss of k; retained-kh mukh forms are held separately rather than assigned by resemblance.'),
(['jib','jibʰā'],['tongue'],'5228','jihvā','Turner gives Bengali jib, Oriya jibha and Bhojpuri jībhⁱ beside MIA jibbhā. Both source b and bha endings belong to this section 1 family.'),
(['hāt'],['arm/ hand'],'14024','hasta','Turner gives Nepali hāt and Bihari/Maithili hāth hand/forearm. The source deaspirated final t is retained with its arm/hand polysemy.'),
(['dãt'],['teeth'],'6152','danta','Turner gives Maithili/Bhojpuri dā̃t tooth, with loss of the nasal stop before the dental and vowel nasalisation. The source unmarked length and plural gloss are preserved.'),
(['peṭ'],['belly'],'8376','*peṭṭa','Turner section 1 explicitly gives Maithili peṭ(h) and Bhojpuri peṭ belly, distinct from poṭṭa and *peḍḍuka lower belly.'),
(['gʰar'],['house'],'4428','ghara','Turner gives Bihari ghar and Maithili ghar/gharo house. The parent is ghara; the deeper debated relationship with gṛha is not skipped.'),
(['gũ'],['feces'],'4225','gūtha','Turner section 1 gives regional gū/guhu and nasalised gũh excrement. The source nasal vowel is retained; the unrelated euphemistic source responses are not pooled with it.'),
(['gām','gāũ'],['village'],'4368','grāma','Turner gives Maithili gām/gā̃o, Bihari gā̃w and Nepali/Bhojpuri gāũ village. Both source m-retaining and nasal-vowel outcomes match the grāma family.')]
a=[]
for forms,glosses,parent,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in forms and r['Gloss'] in glosses:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev+' Published notation and the source uncertainty tag remain unchanged.'))
aud={}
for line in open('data/other/forms/raw_data/selected_surveys_2026/angika-audit.jsonl'):
 row=json.loads(line)
 for reading in row.get('readings',[]):
  ins=reading.get('row',[])
  if len(ins)>7:aud.setdefault(ins[7],[]).extend(reading.get('review',[]))
for x in a:
 notes=aud[x['record']['Source']];assert notes and all(n=='transcription: published notation retained; ambiguous symbols are not assigned guessed phonemic values' for n in notes),(x['record'],notes)
held=[dict(record=r,reason='Retained kh requires discrimination among learned mukha and expressive/related mukhya analyses; the regular mu(h) forms are saved separately.') for r in rs if r['Form']=='mukʰ' and r['Gloss'] in ['mouth','face'] and r['ID'] not in assigned]
(P/'pass013-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n');(P/'pass013-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(a),uncertaintyRetained=True,sources={x['record']['Source']:aud[x['record']['Source']] for x in a}),ensure_ascii=False,indent=2)+'\n')
lines=['# Angika survey: first diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; twelve etyma. No rebuild.','','| # | Angika | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(forms,glosses,parent,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==parent for x in a);label=label.replace('*',r'\*');lines.append(f'| {i} | **{" / ".join(forms)}** ‘{" / ".join(glosses)}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['',f'{len(held)} retained-kh mouth/face records held. All selected source-audit notes were checked; source uncertainty, spelling and locality labels are preserved.','', 'Validation and save status: pending.']
(P/'pass013-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
