import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/belari-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));assigned={r['Form_ID'] for r in read_assignments()};seen={x['record']['ID'] for x in z['accepted']+z['held']}
rules={
'bōḍɨ':('d5528','DEDR5528 gives Kodagu bōḍ- beg and Tulu bēḍuni beg, with bōḍu want. Belari bōḍɨ agrees with the Kodagu long-ō retroflex-stop begging stem, retaining final ɨ. The whole article includes this rounded-vowel branch; it need not match the Tamil head vowel.'),
'ija':('d516','DEDR516 explicitly records Belari (LSB2.1) ija climb down. Survey ija go down (descend) matches the full form and direction of movement; no causative or load-lowering sense is supplied.'),
'ujji':('d710','DEDR710 records Koraga (M.) ujji keep, not Belari: the full prose corrects the loose search language label. Belari ujji is an exact form-and-sense comparison with that neighbouring-language attestation within the be/stay/keep family. It is not claimed as an explicit Belari DEDR citation.'),
'noḍi':('d3783','DEDR3783 explicitly records Belari (LSB2.1) noḍi wash, with Tulu neḍi/niḍiyuni and Parji noḍ-. Survey noḍi matches the complete Belari form; keep it distinct from unrelated noḍi speech and nōḍ- seeing words.'),
'haṇi':('d3888','DEDR3888 gives Tamil paṇi bow/be lowered and Kannada (Hal.) hənuku bend/bow. Homepage discovery additionally finds Kannada haṇi bow/bend explicitly citing3888. Belari haṇi bend is a qualified h-initial nasal-stem comparison. Keep it separate from the distinct Belari haṇɨ say record under3887; similar spellings do not merge the meanings.')}
holds={
'upparsɨ':([],'Tulu climbing forms and relaxed uppar search do not establish the complete Belari upparsɨ climb. The superficially similar uppēruni in710 is cattle copulating with an explicit climb constituent, not evidence for this full form. Morphological analysis remains open.'),
'soccoḍɨ':([],'Tulu throw query supplies unrelated aḍak-, udēv-, pāḍ- and bī- formations; socco form search has no match. No supported whole-form or component analysis yet.'),
'iḷa':([],'Tulu wake search supplies lakkāvuni, while relaxed iḷa matches broader unrelated words and kiḷar rise forms with an extra initial velar. No established correspondence or direct Belari comparison for iḷa wake up yet; preserve source meaning.')}
for r in json.load(open(p/'inventory.json')):
 if r['ID'] in assigned or r['ID'] in seen:continue
 if r['Form'] in rules:
  d,e=rules[r['Form']];z['accepted'].append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=e+' Inheritance versus local transmission remains subject to review.',status='pending-review'))
 elif r['Form'] in holds:
  c,e=holds[r['Form']];z['held'].append(dict(record=r,reason=e,candidates=c))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');a=json.load(open(p/'primary-evidence.json'));ids={x['parent'] for x in z['accepted']};a.update({r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in ids and r['Status']!='duplicate-excluded'});(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');print(len(z['accepted']),len(z['held']))
