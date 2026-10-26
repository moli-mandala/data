import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'tamilnadu-ia-records.json').read_text());assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['paṇi','pani'],'water','8082','pānīya','Turner gives Marathi/Gujarati pāṇī and Konkani pāṇi water under pānīya. Both source n/ṇ readings fit this regional water family; their unmarked vowel length is preserved.'),
(['kan','kə̄n'],'ear','2830','karṇa','Turner gives Marathi/Gujarati kān and Konkani kānu ear through MIA kaṇṇa. The source central-vowel Saurashtra form and short-vowel Katkari form are retained in this family assignment.'),
(['dola','dɔlo'],'eye','6582','dola','Turner gives Prakrit ḍōla eye and Marathi ḍoḷā eye under dola, beside the swinging/litter senses. The source d/l notation is not retroflexed silently; the eye sense and regional dol- family support the link.'),
(['kesa','khɛ̄s'],'hair','3471','keśa','Turner gives Marathi kes/kẽs and Gujarati keś hair beside MIA kesa. The source kesa and Saurashtra khɛs are assigned to this family with the aspiration and vowel quality explicitly preserved.'),
(['at','hə̄t'],'hand','14024','hasta','Turner gives Marathi hāt and Gujarati hāth hand through MIA hattha. The source Katkari at lacks h and Saurashtra hə̄t has a central vowel; those reported shapes are preserved in the regional hand-family assignment.'),
(['nak','nə̄k'],'nose','6909','*nakka','Turner gives Marathi/Gujarati nāk nose under *nakka. The source central-vowel and short-vowel variants retain the diagnostic k and nose meaning.'),
(['jīv'],'tongue','5228','jihvā','Turner includes the MIA jivhā/jibbhā tongue family and western jībh forms under jihvā. Saurashtra jīv is assigned here with its v outcome preserved; the differently transcribed Katkari Ɉip is held for further discrimination.'),
(['dat','də̄t'],'tooth','6152','danta','Turner gives Marathi/Gujarati dā̃t and Konkani dāntu tooth under danta. Source unmarked nasalisation and the Saurashtra central vowel remain unchanged.'),
(['dut','dūd'],'milk','6391','dugdha','Turner gives duddha in MIA, Marathi/Gujarati dūdh and Konkani dūd milk; dut is also explicitly attested in the wider family. Source final t/d outcomes and vowel lengths are preserved.'),
(['gər','ghɛ̄r'],'house','4428','ghara','Turner gives Marathi/Gujarati ghar house under ghara, with regional vowel variation also in the addendum. Source ə/ɛ̄ and aspiration remain unchanged; no claim of a direct gṛha link is made.'),
(['don'],'two','6648','dva','Turner explicitly gives Marathi don and Konkani dōni two after analogical MIA doṇṇi. This is not the unrelated donnā/don land-between-rivers entry 6598.'),
(['tin','tīnu'],'three','5994-3','trīṇi','Turner section 3 gives Marathi tīn and Konkani tīni three through MIA tiṇṇi. Both source t-initial forms select the trīṇi subsection rather than trayaḥ.'),
(['car','cə̄ru'],'four','4655-2','catvāri','Turner section 2 gives Marathi cār and Konkani cyāri four. Both source c-r forms fit this catvāri series; the Saurashtra central vowel and final u are retained.'),
(['pãc','pə̄nju'],'five','7655','pañca','Turner gives Marathi pā̃c/Konkani pā̃ñca and wider panj reflexes under pañca. The source Katkari nasal-vowel c form and Saurashtra nj form are assigned to this family without normalising their endings.'),
(['ce'],'six','12803-3','*kṣaṭ','Turner section 3 gives regional che/cha and Marathi sahā/Konkani sa six through MIA cha. Katkari ce belongs to this branch; the unexplained Saurashtra sɔ̄tu is excluded.')]
a=[]
for fs,g,pa,la,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in fs and r['Gloss']==g:a.append(dict(record=r,parent=pa,kind='reflex',citation=f'CDIAL[{pa}];'+r['Source'],evidence=ev+' Source audit checked; comparative-table uncertainty and source language/dialect mapping remain unchanged.'))
aud={}
for line in open('data/other/forms/raw_data/census_nepal_2026/tamil-nadu-audit.jsonl'):
 z=json.loads(line)
 for rd in z.get('readings',[]):
  ins=rd.get('installed',[])
  if len(ins)>7:aud.setdefault(ins[7],[]).append(rd)
allowed={'dialect mapping: Katkari follows chapter p. 326 despite contradictory Dravidian table heading','gloss/transcription: published comparative table contains editorial inconsistencies; attested source pairing preserved','transcription: embedded image restored; original orthography retained'}
for x in a:
 ar=aud[x['record']['Source']];assert any(z['installed'][3]==x['record']['Gloss'] for z in ar);assert all(all(n in allowed for n in z['review']) for z in ar)
reasons={'gusto':'Fire response requires independent lexical research; not a transparent agni-family reflex.','ittāu':'Resolve immediate history of this fire response, including contact alternatives.','Ɉip':'The unusual initial transcription and final p require additional source/language evidence before choosing a jihvā link.','diyɔ':'Resolve the specific two-stem formation rather than importing the Katkari don analysis.','sɔ̄tu':'Source six response has unexplained t/u for the ordinary six branches and could involve an elicitation inconsistency; no forced link.'}
held=[dict(record=r,reason=reasons[r['Form']]) for r in rs if r['ID'] not in assigned and r['Form'] in reasons and r['Gloss'] in ['fire','tongue','two','six']]
(P/'pass020-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n');(P/'pass020-source-audit.json').write_text(json.dumps({x['record']['Source']:aud[x['record']['Source']] for x in a},ensure_ascii=False,indent=2)+'\n')
lines=['# Tamil Nadu survey: Katkari and Saurashtra','',f'{len(a)} records / {len(a)} reflex links; fifteen etyma. Katkari follows the existing chapter-based mapping despite the table-heading contradiction. No rebuild.','','| # | Survey forms | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(fs,g,pa,la,ev) in enumerate(groups,1):
 la=la.replace('*',r'\*');n=sum(x['parent']==pa for x in a);lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ ({n}) | **{la}**, CDIAL **{pa}** | {ev} |')
lines+=['',f'{len(held)} held records:']+[f'- **{x["record"]["Form"]}** ‘{x["record"]["Gloss"]}’: {x["reason"]}' for x in held]+['','All selected source audit notes checked; uncertainty tags and phonetic transcription retained. These links are regional family analyses, not a claim that Turner prints the exact modern survey spellings.','', 'Validation and save status: pending.'];(P/'pass020-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')
