import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'danuwar-records.json').read_text());assigned={r['Form_ID'] for r in read_assignments()};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['ā̃kh'],['eye'],'43','akṣi','Turner gives Bihari/Maithili ā̃khi, Hindi ā̃kh and Nepali ā̃kho eye; the nasal vowel and aspirated velar match the akṣi > akkhi family.'),
(['kān'],['ear'],'2830','karṇa','Turner explicitly gives Nepali, Maithili and Bhojpuri kān ear, through MIA kaṇṇa with cluster simplification and compensatory length.'),
(['nāk','nākh'],['nose'],'6909','*nakka','Turner gives Nepali/Maithili/Bhojpuri nāk and Kumauni nākh nose under *nakka. Both aspirated and unaspirated source outcomes are represented in the regional family; the deeper relation with nas is not used as a direct parent.'),
(['dā̃t','d̺ā̃t̺','dāt'],['teeth'],'6152','danta','Turner gives Nepali/Maithili/Bhojpuri dā̃t tooth, with dental-cluster simplification. Source dental diacritics and nasalisation differences are retained; the plural English elicitation gloss does not create another etymon.'),
(['hāt','hāt̺','hāth'],['arm/ hand'],'14024','hasta','Turner gives Nepali hāt and Bihari/Maithili hāth hand/forearm. Both hāt and hāth outcomes fit the regional hasta family, with the complete source arm/hand meaning preserved.'),
(['jih'],['tongue'],'5228','jihvā','Turner explicitly gives Maithili jihᵃ/jīhᵃ and MIA jihā/jīhā tongue under section 1. The source jih has the expected loss of final vowel; the other source tongue forms are not silently grouped with it.'),
(['ghar'],['house'],'4428','ghara','Turner gives Nepali and Bihari ghar house beside Maithili ghar/gharo. Link to the established ghara node; Turner considers its deeper derivation from gṛha phonetically difficult.'),
(['kes'],['hair'],'3471','keśa','Turner gives Nepali and Maithili kes hair beside Pali/Prakrit kesa. The source hair sense and sibilant fit this inherited family.'),
(['gor'],['leg'],'4272','*goḍḍa','Turner gives Maithili goṛ and Bhojpuri gōṛ foot/leg beside Nepali goṛo. The source writes r; no retroflex sign is silently inserted into the transcription.')]
a=[]
for forms,glosses,parent,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in forms and r['Gloss'] in glosses:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev+' Existing language/dialect mapping uncertainty remains: the survey uses broad Danuwar labels and the installed Done/East distinction is geographical.'))
# Source-level uncertainty is checked record by record, not assumed to be a reading problem.
audit={}
for line in open('data/other/forms/raw_data/census_nepal_2026/danuwar-audit.jsonl'):
 row=json.loads(line)
 for reading in row.get('readings',[]):
  ins=reading.get('installed',[])
  if len(ins)>7:audit.setdefault(ins[7],[]).extend(reading.get('review',[]))
for x in a:
 notes=audit[x['record']['Source']]
 assert notes and all(n.startswith('dialect mapping:') for n in notes),(x['record'],notes)
held=[dict(record=r,reason='mukha 10158 has regular mu(h) reflexes; mukhya 10174 discusses expressive -kkh- alternatives. Retained kh here needs source-specific distinction from a learned form before an inherited parent is selected.') for r in rs if r['Form']=='mukh' and r['Gloss'] in ['mouth','face'] and r['ID'] not in assigned]
(P/'pass009-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
(P/'pass009-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(a),onlyDialectMappingUncertainty=True,sources={x['record']['Source']:audit[x['record']['Source']] for x in a}),ensure_ascii=False,indent=2)+'\n')
lines=['# Danuwar: first diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; nine etyma. No rebuild.','','| # | Danuwar | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(forms,glosses,parent,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==parent for x in a);label=label.replace('*',r'\*');lines.append(f'| {i} | **{" / ".join(forms)}** ‘{" / ".join(glosses)}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['',f'{len(held)} mukh mouth/face records remain held. All selected audit readings were checked: their uncertainty notes concern geographic language/dialect assignment only. Those labels and uncertainty tags remain unchanged. Source locations and distinct attestations are preserved.','', 'Validation and save status: pending.']
(P/'pass009-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
