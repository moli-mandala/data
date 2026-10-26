import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/belari-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));assigned={r['Form_ID'] for r in read_assignments()};seen={x['record']['ID'] for x in z['accepted']+z['held']}
rules={
'ijḍe':('d583','DEDR583 explicitly records Belari (LSB2.2) ijḍe mortar alongside Tulu ujiḍe/jiḍḍe small mortar. The survey matches the complete Belari form and sense. The article includes both mortars and pestles; its displayed head need not designate the same implement.'),
'ijgerɨ':('d583','DEDR583 explicitly records Belari (LSB2.2) ijgeri pestle alongside Tulu ujjērů/ujverů. Survey ijgerɨ matches the same-language full formation with final ɨ versus i; keep it distinct from ijḍe mortar while linking both to the shared article.'),
'gaḷde':('d1355','DEDR1355 explicitly records Belari (LSB2.3) gaḷde field in the paddyfield family, beside Kannada gar̤de/garde/gadde. The full form matches; retain the broader survey field gloss. The questioned Sanskrit mud comparison is not an immediate donor claim.')}
holds={
'kattige':(['d1204'],'DEDR1204 supplies katti/katte knife across southern languages but does not explain final -ge in Belari kattige. Exact relaxed form discovery only finds unrelated comb material. Full article read; retain complete source form pending morphological evidence.'),
'maḷgɨ':(['d4719'],'Medicine family DEDR4719 has Tulu mardů and Kannada mardu/maddu, but Belari maḷgɨ has an unexplained lateral-velar cluster. Form search finds unrelated lying/decreasing/cloud words. No justified correspondence or whole-form medicine analysis yet.'),
'debirɨ':(['d5485'],'DEDR5485 has Tulu bedůrů and Kannada bidiru/bidaru bamboo. Belari debirɨ may invite a b-d metathesis analysis, but this is not established by the full article or the current source evidence. Preserve as a candidate, not a confirmed link.'),
'hula':([],'Tulu forest discovery supplies kāna/kāḍů/pāḍi/malɛ; relaxed hula search yields other tree names and unrelated words, not a supported whole-form forest match. Do not turn a generic forest gloss into a particular tree or field interpretation without evidence.'),
'sāgo':([],'Tulu pig search supplies pañji and unrelated compound hits; relaxed sāgo search supplies cultivation/plant material. No supported whole-form pig cognate or borrowing route yet. No inference of absence beyond these queries.')}
for r in json.load(open(p/'inventory.json')):
 if r['ID'] in assigned or r['ID'] in seen:continue
 if r['Form'] in rules:
  d,e=rules[r['Form']];z['accepted'].append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=e+' Inheritance versus local transmission remains subject to review.',status='pending-review'))
 elif r['Form'] in holds:
  c,e=holds[r['Form']];z['held'].append(dict(record=r,reason=e,candidates=c))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');a=json.load(open(p/'primary-evidence.json'));ids={x['parent'] for x in z['accepted']};a.update({r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in ids and r['Status']!='duplicate-excluded'});(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');print(len(z['accepted']),len(z['held']))
