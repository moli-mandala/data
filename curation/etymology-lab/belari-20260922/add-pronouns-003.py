import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/belari-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));assigned={r['Form_ID'] for r in read_assignments()};seen={x['record']['ID'] for x in z['accepted']+z['held']}
rules={
'ēṇɨ':('d5160','DEDR5160 gives Tulu ēnů/yēnu I, alongside the yānu series. Belari ēṇɨ preserves long ē and a nasal with final high unrounded vowel; the source retroflex nasal differs from the dictionary Tulu n and remains a qualification, not a transcription repair.'),
'eŋkḷo':('d5154','DEDR5154 gives Tulu eṅkuḷů we (exclusive), with Kodagu eŋga we; the full article also includes inclusive uses in other languages. Belari eŋkḷo matches the nasal-velar-lateral plural formation with a reduced medial vowel and final o. The installed inclusive and exclusive records share the same form. Bhat1971 p.123 lists one undifferentiated we form; retain both source records without claiming two independently attested clusivity forms.'),
'āyi':('d1','DEDR1 gives Tulu āye that man and remote demonstratives across the family. Belari āyi remote masculine singular matches the long-ā glide formation with final i versus e; do not infer a separate lexical man constituent from the English gloss.'),
'ākḷɨ':('d1','DEDR1 gives Tulu ākuḷu those persons. Belari ākḷɨ remote animate plural agrees in ā-k-lateral with medial vowel reduction and final ɨ. This is the inflected remote pronoun within the whole article, not a proposed lexical compound.'),
'adɨ':('d1','DEDR1 includes Telugu adi that woman or thing, Kodagu adï that thing/those things and widespread ad forms. Belari adɨ is a remote dental-stop demonstrative. Preserve its three separate source records (female singular, inanimate singular and inanimate plural); the article supports shared nonmasculine demonstrative morphology, not three distinct stems.')}
for r in json.load(open(p/'inventory.json')):
 if r['ID'] in assigned or r['ID'] in seen or r['Form'] not in rules:continue
 d,e=rules[r['Form']];z['accepted'].append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=e+' Inheritance versus local transmission remains subject to review.',status='pending-review'))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
a=json.load(open(p/'primary-evidence.json'));ids={x['parent'] for x in z['accepted']};a.update({r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in ids and r['Status']!='duplicate-excluded'});(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
print(len(z['accepted']))
