import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'sikkim-records.json').read_text());assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['āgo','āgō','āgī'],'fire','55','agni','Turner explicitly gives Nepali āgo fire and regional āgi/āgī beside MIA aggi. The o- and i-final responses belong to this family; retained-cluster agnī is held separately.'),
(['pānī'],'water','8082','pānīya','Turner gives Nepali pāni water under pānīya. Source vowel lengths are retained; no unrecorded distinction between the locality readings is imposed.'),
(['gʰar','gʰār'],'house','4428','ghara','Turner explicitly gives Nepali ghar house under ghara. The source long-vowel variant is preserved. The deeper debated relation to gṛha is not skipped.'),
(['kān'],'ear','2830','karṇa','Turner gives Nepali kān ear from the karṇa > kaṇṇa family. The three source locality attestations share that regular cluster simplification.'),
(['ā̃kʰā','ā̃kʰī'],'eye','43','akṣi','Turner gives Nepali ā̃kho eye and ā̃khi evil eye. The source ā̃kha/ā̃khi responses continue akṣi through akkhi, with source endings and eye gloss retained. The extra-n response is held.'),
(['kes'],'hair','3471','keśa','Turner gives Nepali kes hair alongside MIA kesa. This regular s-form is linked; source keś responses are held for the status of their retained sibilant.'),
(['bāl'],'hair','11572','vāla','Turner explicitly gives Nepali bāl hair under vāla. The b onset and long vowel support this separate hair family.'),
(['hāt','hātʰ'],'hand','14024','hasta','Turner gives Nepali hāt hand/forearm and neighbouring Bihari/Kumaoni hāth. Both source aspiration variants fit hasta > hattha; repeated hand elicitation records are kept distinct.'),
(['nāk','nākʰ'],'nose','6909','*nakka','Turner gives Nepali nāk and Kumaoni nākh nose under *nakka. Source aspiration is preserved; the regional k/kh responses are assigned to this nose family.'),
(['jībro'],'tongue','5228','jihvā','Turner explicitly gives Nepali jib(h)ro tongue under jihvā. The source r-bearing stem matches that attested regional extension; no independent suffix etymology is invented.'),
(['dā̃t'],'tooth','6152','danta','Turner explicitly gives Nepali dā̃t tooth. The source nasal vowel and final dental match this family; the anomalously aspirated response is held.'),
(['dūdʰ','dūd'],'milk','6391','dugdha','Turner gives Nepali dud(h)/dut milk under dugdha through MIA duddha. These nominal milk responses are distinct from the survey verbal elicitation glossed milk.'),
(['dʰū̃ā'],'smoke','6849','dhūma','Turner section 1 gives Nepali dhuwā̃ smoke beside regional dhuā̃. The source dhũa fits loss of intervocalic m with nasalisation; dhũgo and dhumra are not pooled with it.'),
(['peṭ'],'belly','8376','*peṭṭa','Turner section 1 explicitly gives Nepali peṭ belly under *peṭṭa. This differs from the source bhũri/bhudi alternatives.'),
(['bʰū̃rī'],'belly','9531','*bhuṇḍa','Turner explicitly gives Nepali bhũṛi belly under *bhuṇḍa. The source writes bhũri; its r notation and vowel lengths are preserved rather than silently retroflexed.')]
a=[]
for fs,g,pa,la,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in fs and r['Gloss']==g:a.append(dict(record=r,parent=pa,kind='reflex',citation=f'CDIAL[{pa}];'+r['Source'],evidence=ev+' Source audit checked; published comparative-table uncertainty and locality tags remain unchanged.'))
aud={}
for line in open('data/other/forms/raw_data/census_nepal_2026/sikkim2-audit.jsonl'):
 row=json.loads(line)
 for rd in row.get('readings',[]):
  ins=rd.get('installed',[])
  if len(ins)>7:aud.setdefault(ins[7],[]).append(rd)
for x in a:
 ar=aud[x['record']['Source']];assert any(z['installed'][3]==x['record']['Gloss'] for z in ar);assert all(all(n in ['transcription: embedded image restored; original orthography retained','gloss/transcription: published comparative table contains editorial inconsistencies; attested source pairing preserved'] for n in z['review']) for z in ar)
reasons={'agnī':'Retained gn suggests learned agni; resolve immediate donor and relation separately.','keś':'Retained ś may reflect learned vocabulary or editorial spelling; clarify before treating as inherited kes.','ā̃nkʰā':'Source combines nasal vowel with extra n; retain pending transcription and phonological discrimination.','dā̃tʰ':'Source aspirated tooth response differs from the Nepali dā̃t comparandum; retain for further review.','bʰūdī':'Compare bhuṇḍa 9531 and the alternative *bhudda noted in its addendum; source shape is not resolved.','dʰū̃go':'Unexplained g in the smoke response requires source or lexical discrimination.','dʰūmrā':'Retained mr may belong to dhūmra or learned vocabulary rather than regular dhūma reflex; resolve separately.'}
held=[dict(record=r,reason=reasons[r['Form']]) for r in rs if r['ID'] not in assigned and r['Form'] in reasons and r['Gloss'] in ['fire','hair','eye','tooth','belly','smoke']]
(P/'pass017-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n');(P/'pass017-source-audit.json').write_text(json.dumps({x['record']['Source']:aud[x['record']['Source']] for x in a},ensure_ascii=False,indent=2)+'\n')
lines=['# Sikkim survey: Nepali comparison vocabulary','',f'{len(a)} records / {len(a)} reflex rows; fifteen etyma. All three published localities preserved. No rebuild.','','| # | Nepali | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(fs,g,pa,la,ev) in enumerate(groups,1):
 la=la.replace('*',r'\*');n=sum(x['parent']==pa for x in a);lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({n}) | **{la}**, CDIAL **{pa}** | {ev} |')
lines+=['',f'{len(held)} records held. Verbal milk responses were not included in nominal milk. All selected source audit notes were checked; no source uncertainty tags were removed.','', 'Validation and save status: pending.'];(P/'pass017-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
