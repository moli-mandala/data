import json,csv
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
records=json.loads((P/'darai-records.json').read_text())
assigned={r['Form_ID'] for r in read_assignments()}
assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('tel','oil','5958','taila','Turner lists Pali/Pk tēla and Nepali, Bihari and Maithili tel oil; monophthongal e and loss of the final vowel match the regional inherited family.'),
('agi','fire','55','agni','Turner gives Pk aggi and Old Bengali, Oriya and Bihari āgi fire. The simplified gn cluster and retained final i match this eastern family.'),
('ʦar','four','4655-2','catvāri','The cār/cāri numeral series belongs specifically to section 2 catvāri, with MIA *cayāri and Apabhramsha cāri; compare Nepali cār and Maithili cāri. The source c is rendered ʦ by its existing transcription profile.'),
('pãʦ','five','7655','pañca','Compare Nepali and Maithili pā̃c; nasalisation replaces the nasal before the retained affricate, and the final vowel is lost.'),
('dui','two','6648','dva','Turner explicitly gives Nepali, Bihari, Maithili and Bhojpuri dui, continuing the duv-/dui series.'),
('ek','one','2462-2','*ekka','The retained k belongs to the *ekka section 2 rather than the single-k MIA ega/ea series. Turner gives Nepali and Maithili ek; his alternatives of emphatic gemination or early MIA Sanskrit replacement remain unresolved.'),
('gʰiu','ghee','4501','ghṛta','Pk ghia and Nepali/Assamese/Awadhi ghiu provide the regular loss of medial t and vocalisation of syllabic r; aspiration and the iu sequence are retained.'),
('dʰan','paddy','6778','dhānya','The paddy sense matches dhānya and Pk dhaṇṇa growing rice, with Nepali/Bihari/Maithili dhān; it is distinct from dhānā roasted grain.'),
('bar','hair','11572','vāla','Turner explicitly lists Bhojpuri bār hair beside Nepali bāl, supporting initial v > b and the regional l/r correspondence. The homonymous fruit record is excluded.'),
('muḍʰ','head','10247','mūrdhan','Pk muḍḍha and Nepali muṛ(h) head support the retroflex aspirated stop. Turner discusses muṇḍa contamination especially for unaspirated forms; the aspiration here supports mūrdhan directly.')]
a=[]
for form,gloss,parent,label,ev in groups:
 for r in records:
  if r['Form']==form and r['Gloss']==gloss and r['ID'] not in assigned:
   a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];dhakal2011darai[pp. 42–43]',evidence=ev+' Dhakal states explicitly on p. 42 that vowel length is not contrastive; no missing length mark is inferred or edited.'))
held=[dict(record=r,reason='The prastara family is supported, but Turner marks some neighboring short-a stone forms as loans. The Darai immediate route needs stronger evidence before assigning inheritance or a donor.') for r in records if r['Form']=='patʰər' and r['Gloss']=='stone' and r['ID'] not in assigned]
(P/'pass001-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
print(len(a),'accepted',len(held),'held')
lines=['# Darai: first diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and assignment rows; ten etyma. No rebuild.','','| # | Darai | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(form,gloss,parent,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==parent for x in a)
 lines.append(f'| {i} | **{form}** ‘{gloss}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['','Dhakal pp. 42–43 explicitly describe noncontrastive vowel length. Source records and homonyms are preserved. All assignments are reflex links. Stone remains held over its contact history.','', 'Validation and save status: pending.']
(P/'pass001-review.md').write_text('\n'.join(lines)+'\n')
