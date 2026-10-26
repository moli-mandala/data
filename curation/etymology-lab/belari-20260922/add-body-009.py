import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/belari-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));assigned={r['Form_ID'] for r in read_assignments()};seen={x['record']['ID'] for x in z['accepted']+z['held']}
rules={
'bergi':('d5488','DEDR5488 explicitly records Belari (LSB2.2) bergi back, beside Tulu beri and Kannada ben/bennu. Survey bergi matches the complete Belari form and body-part sense; no need to invent a derivation from the similarly related spatial expressions in the article.'),
'hodkuḷu':('d4460','DEDR4460(a) explicitly records Belari (LSB2.2) hodkuḷu navel. The full form, including the medial dk sequence, exactly matches the survey. Preserve it rather than normalizing to Kannada hokkuḷu or requiring the head form to display this sequence.')}
holds={
'oḍivu':(['d3748'],'Tulu blood discovery supplies netterů, but full DEDR3748 has no comparandum explaining Belari oḍivu. Relaxed form search returns unrelated words. This is an unresolved blood term; do not force the dominant local blood family onto it.'),
'gōke':(['d1253'],'DEDR1253 includes Tulu kekki/kekkilů and Koraga gekkili neck. Belari gōke differs in vowel quantity and quality and consonant length; the source evidence has not established those correspondences. Exact relaxed form discovery found no matching dictionary form. Keep as an open candidate.'),
'jaŋgɨ':(['d2286'],'DEDR2286 supplies Kannada jaṅge and Telugu jaṅga stride, and explicitly says the listed words probably derive from Sanskrit jaṅghā lower leg. Belari jaŋgɨ foot or leg is suggestive but lacks a verified immediate donor route. Do not attach as inherited merely because the stride forms share this article; do not bypass the donor with a Sanskrit ancestry link.')}
for r in json.load(open(p/'inventory.json')):
 if r['ID'] in assigned or r['ID'] in seen:continue
 if r['Form'] in rules:
  d,e=rules[r['Form']];z['accepted'].append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=e+' Inheritance versus local transmission remains subject to review.',status='pending-review'))
 elif r['Form'] in holds:
  c,e=holds[r['Form']];z['held'].append(dict(record=r,reason=e,candidates=c))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');a=json.load(open(p/'primary-evidence.json'));ids={x['parent'] for x in z['accepted']};a.update({r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in ids and r['Status']!='duplicate-excluded'});(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');print(len(z['accepted']),len(z['held']))
