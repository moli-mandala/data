import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'westbengal-records.json').read_text());assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['jūī'],'fire','5300','jyotis','Turner gives Assamese zūi and Bengali jũi fire under jyotis. This matches the Rajbanshi response. The dyuti 6606 addendum also gives Assamese jui fire; that competing analysis remains explicitly open, following the earlier reviewed family choice.'),
(['jɔl'],'water','5155','jala','Turner gives Bengali jal water beside MIA jala. The source ɔ is preserved. The Rajbanshi form is assigned by the same regional correspondence, without claiming Turner printed this survey attestation.'),
(['pānī'],'water','8082','pānīya','Turner explicitly gives Nepali pāni water under pānīya. The survey final vowel length is preserved.'),
(['pɛṭ','peṭ'],'belly','8376','*peṭṭa','Turner section 1 gives Bengali/Nepali peṭ belly. The Rajbanshi ɛ and Nepali e remain as recorded; the compound tɔl peṭ is not treated as one inherited stem.'),
(['kān'],'ear','2830','karṇa','Turner gives Bengali kāṇ and Nepali kān ear under karṇa through kaṇṇa. The source dental n is preserved across Bengali, Rajbanshi and Nepali.'),
(['ā̃kʰī'],'eye','43','akṣi','Turner explicitly gives Nepali ā̃khi under akṣi through akkhi. The survey eye sense and final ī are retained; the distinct Bengali/Rajbanshi cok family is held for its borrowing route.'),
(['cūl'],'hair','4883','cūḍa','Turner section 1 gives Bengali cul hair of the head from the cūḍa/cūla topknot family. The source Bengali/Rajbanshi cūl is assigned to this family; capital-C spelling is held for notation review.'),
(['bāl'],'hair','11572','vāla','Turner explicitly gives Nepali bāl hair under vāla. This source response is distinct from alternative kes hair.'),
(['kes'],'hair','3471','keśa','Turner explicitly gives Nepali kes hair beside MIA kesa. The source s-series form matches directly.'),
(['hāt'],'hand','14024','hasta','Turner gives Bengali hāt arm and Nepali hāt hand/forearm under hasta. Source hand senses agree with this regional hand/arm family; Rajbanshi is assigned by the same correspondence.'),
(['nāk'],'nose','6909','*nakka','Turner explicitly gives Bengali and Nepali nāk nose under *nakka. The identical Rajbanshi source form belongs to the same regional family.'),
(['jīb','jībʰā','jībro'],'tongue','5228','jihvā','Turner gives Bengali jib, eastern jibha and Nepali jib(h)ro tongue under jihvā. Each source form is retained, including the Nepali r extension and Rajbanshi bhā ending.'),
(['dā̃t','dāt'],'tooth','6152','danta','Turner gives Bengali/Nepali dā̃t tooth. The Rajbanshi source lacks marked nasalisation; that notation remains unchanged in assigning the same long-vowel dental family.'),
(['dūd','dūdʰ'],'milk','6391','dugdha','Turner gives Bengali/Nepali dud(h) milk from dugdha through duddha. The Rajbanshi dūdh response matches; verbal milk responses and khāudā are excluded.'),
(['gʰar'],'house','4428','ghara','Turner explicitly gives Nepali ghar house under ghara. This does not skip to the debated deeper gṛha relationship.'),
(['bāṛī','bārī'],'house','11480','vāṭa','Turner gives Bengali bāṛi garden and regional vāṭa-family enclosure/habitation senses, including Konkani vāḍo habitation. The Bengali/Rajbanshi house responses are analysed as the regional homestead sense of this family; this semantic inference and source r/ṛ distinction are retained.')]
a=[]
for fs,g,pa,la,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in fs and r['Gloss']==g:a.append(dict(record=r,parent=pa,kind='reflex',citation=f'CDIAL[{pa}];'+('CDIAL[6606];' if pa=='5300' else '')+r['Source'],evidence=ev+' Source audit checked; published spelling, language/dialect assignment and uncertainty tag remain unchanged.'))
aud={}
for line in open('data/other/forms/raw_data/more_surveys_2026/west-bengal-audit.jsonl'):
 z=json.loads(line)
 for rd in z.get('readings',[]):
  ins=rd.get('row',[])
  if len(ins)>7:aud.setdefault(ins[7],[]).append(rd)
for x in a:
 ar=aud[x['record']['Source']];assert any(z['row'][3]==x['record']['Gloss'] for z in ar);assert all(all(n=='transcription/gloss: source attestation preserved; inconsistent notation or prompt meaning is not silently emended' for n in z['review']) for z in ar)
reasons={'āgūn':'Resolve the n-bearing fire stem rather than pooling it with simple āg.','āgī - agnī - āgo':'Mixed response combines inherited and potentially learned stems; do not assign one relation to the entire string.','cok':'Turner marks the Bengali cakṣus family probably borrowed from Prakrit. Resolve the immediate donor rather than asserting direct inheritance.','cokū':'Resolve the regional extension and the probable Prakrit borrowing route of the eye family.','Cūl':'Capital-C notation differs from cūl; resolve its source interpretation before pooling.','tɔl peṭ':'Multiword lower-belly expression needs component analysis.','kʰāūdā':'This milk response lacks a supported simple nominal milk etymology; preserve the source pairing.'}
held=[dict(record=r,reason=reasons[r['Form']]) for r in rs if r['ID'] not in assigned and r['Form'] in reasons and r['Gloss'] in ['fire','eye','hair','belly','milk']]
(P/'pass019-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n');(P/'pass019-source-audit.json').write_text(json.dumps({x['record']['Source']:aud[x['record']['Source']] for x in a},ensure_ascii=False,indent=2)+'\n')
lines=['# West Bengal survey: Bengali, Rajbanshi and Nepali','',f'{len(a)} records / {len(a)} reflex rows, sixteen etyma. No rebuild.','','| # | Survey forms | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(fs,g,pa,la,ev) in enumerate(groups,1):
 la=la.replace('*',r'\*');n=sum(x['parent']==pa for x in a);lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({n}) | **{la}**, CDIAL **{pa}** | {ev} |')
lines+=['',f'{len(held)} records held:']+[f'- **{x["record"]["Form"]}**: {x["reason"]}' for x in held]+['','All selected source audit notes checked. Verbal milk elicitation responses are excluded from the nominal milk group.','', 'Validation and save status: pending.'];(P/'pass019-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
