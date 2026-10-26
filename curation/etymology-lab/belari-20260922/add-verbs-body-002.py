import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/belari-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));inv=json.load(open(p/'inventory.json'));assigned={r['Form_ID'] for r in read_assignments()};seen={x['record']['ID'] for x in z['accepted']+z['held']}
rules={
'mī':('bathe','d4878','DEDR4878 gives Kannada mī/mīyu and Tulu mīpini take a bath, with Parji mī-. Belari mī preserves the long-vowel bathing stem. The source bathe gloss does not imply the causative interpretation of the separately listed mīpāvuni.'),
'būḷɨ':('fall','d5430','DEDR5430 explicitly gives Koraga (M.) būḷu and Kodagu būḷ- fall, alongside Tulu būruni. Belari būḷɨ agrees in the long ū and retroflex lateral, differing in the final-vowel transcription; it need not match the Tamil head form.'),
'hugelɨ':('shoulder','d4172','DEDR4172 explicitly records Belari (LSB2.3) hugeli shoulder, beside Tulu pugelů/puggelů. Belari hugelɨ matches the same-language h-g-l attestation with final ɨ versus i. Search labels spill the Belari material into Tulu; this proposal follows the full primary article, not that label.'),
'argi':('rice','d215','DEDR215 explicitly records Belari (LSB2.2) argi rice. This matches the full source form and sense; it is not Tulu ari with an analyst-inserted stop. The search output mislabels this Belari comparandum as Tulu, corrected here from the full article.'),
'māḷɨ':('do','d4758','DEDR4758 gives Tulu maḷpuni do/make and Telugu mal(u)cu make/fashion. Belari māḷɨ agrees in the m-retroflex-lateral stem and do sense, but has marked long ā and no -puni. This is a qualified stem-level comparison; the source does not establish a Tulu borrowing route.'),
'haṇdɨ':('fruit','d4004','DEDR4004 contains the whole ripe-fruit family, including Telugu paṇḍu and Tulu parndů, as well as Kannada paṇ/paṇṇu. Homepage discovery also finds Kannada haṇ/haṇṇu explicitly cited to DEDR4004 in Uchida–Rajapurohit p.904. Belari haṇdɨ combines initial h with a nasal-stop fruit stem; the ṇd and final ɨ are preserved, not normalized to the Kannada form. The Sanskrit phala comparison does not establish a donor.'),
'huḷe':('river','d4318','DEDR4318 gives Kannada por̤e, Kodagu poḷe and Malayalam pur̤a river. Homepage discovery finds Kannada hoḷe stream/small river in Uchida–Rajapurohit p.938, explicitly citing DEDR4318. Belari huḷe is a qualified h-initial lateral-stem comparison, with u versus the Kannada/Kodagu o; it is not an exact primary DEDR attestation.')}
for r in inv:
 if r['ID'] in seen or r['ID'] in assigned:continue
 if r['Form'] in rules:
  g,d,e=rules[r['Form']];assert r['Gloss']==g
  z['accepted'].append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=e+' The proposed link uses the whole DEDR article; inheritance versus local transmission remains subject to review.',status='pending-review'))
 elif r['Form']=='hoḍalɨ':
  z['held'].append(dict(record=r,reason='DEDR586 has Kannada oḍal and Tulu uḍalů/oḍālů belly, a plausible medial-stem match. Belari initial h is not present in the full article comparanda; preserve the source form and seek a direct Belari attestation or evidence for this prothesis before proposing the link.',candidates=['d586']))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
a=json.load(open(p/'primary-evidence.json'));ids={x['parent'] for x in z['accepted']};a.update({r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in ids and r['Status']!='duplicate-excluded'});(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
print(len(z['accepted']),len(z['held']))
