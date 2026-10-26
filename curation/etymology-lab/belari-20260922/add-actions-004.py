import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/belari-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));assigned={r['Form_ID'] for r in read_assignments()};seen={x['record']['ID'] for x in z['accepted']+z['held']}
rules={
'ubi':('spit','d636','DEDR636 gives Tulu (B-K.) ubi spit, as well as ubbiyuni and Koraga ubbi. Belari ubi matches the whole simple verb; dictionary variants with gemination or different medial consonants do not require changing its spelling.'),
'kārɨ':('vomit','d1477','DEDR1477 gives Tulu kāruni and Kodagu/Kota kār- vomit, alongside Malayalam kāruka. Belari kārɨ preserves the long vowel and rhotic stem with final ɨ. No causative meaning is inferred from the separate causative formations in the article.'),
'haṇɨ':('say','d3887','DEDR3887 explicitly gives Belari (LSB2.3) haṇi say. Belari haṇɨ matches the same-language h-retroflex-nasal stem with final ɨ versus i; the source say gloss is retained, rather than importing the command/send senses of other article members.'),
'hijinɨ':('ant','d4137','DEDR4137 explicitly gives Belari (LSB2.3) hijini ant beside Tulu pijinů. The survey hijinɨ matches the full same-language form with final ɨ versus i. This is the specific ant family, not the unrelated hoyda or generic insect search matches.'),
'hūju':('smear','d4352','DEDR4352 gives Tulu pūjuni smear/rub/daub and the Tamil pūcu family. Belari hūju preserves the long ū and j, with initial h corresponding to Tulu p as described by Bhat1971 pp.120–121. This supports the family comparison but not a uniquely established borrowing route; the Sanskrit comparison in the article is not a donor assertion.'),
'hoḷtɨ':('time','d4559','DEDR4559 includes Tulu poḷtu time/daylight/sun and Kannada hottu. Belari hoḷtɨ preserves the lateral-stop stem and time sense, with h corresponding to Tulu p and final ɨ versus u. Bhat1971 p.121 compares these time forms; the broader article also contains sun words without requiring a changed survey meaning.'),
'hoy':('beat','d4534','DEDR4534 gives Kannada poy beat and Kodagu poyy beat, alongside Kota poy- strike with the hand. Homepage discovery also finds Kannada hoyyu beat explicitly citing DEDR4534. Belari hoy matches this h-initial beating stem. Keep it distinct from the unrelated pouring family DEDR4407, despite homonymous hoy search results.')}
for r in json.load(open(p/'inventory.json')):
 if r['ID'] in assigned or r['ID'] in seen:continue
 if r['Form'] in rules:
  g,d,e=rules[r['Form']];assert r['Gloss']==g;z['accepted'].append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=e+' The proposal links to the whole article; inheritance versus local transmission remains subject to review.',status='pending-review'))
 elif r['Form']=='ū' and r['Gloss']=='blow (with mouth)':
  z['held'].append(dict(record=r,reason='DEDR741 (ūtu/ūdu/ūbu) and DEDR751 (ūr-/urpu) both contain mouth-blowing verbs. Belari ū lacks the consonants distinguishing them. Full articles read; retain both candidates until a paradigm or direct comparative source establishes which family explains this short form.',candidates=['d741','d751']))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');a=json.load(open(p/'primary-evidence.json'));ids={x['parent'] for x in z['accepted']};a.update({r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in ids and r['Status']!='duplicate-excluded'});(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');print(len(z['accepted']),len(z['held']))
