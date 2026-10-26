import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/belari-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));assigned={r['Form_ID'] for r in read_assignments()};seen={x['record']['ID'] for x in z['accepted']+z['held']}
rules={
'marjedɨ':('d4715','DEDR4715 explicitly records Belari (LSB2.2) marjedi grandchild, with Koraga marje grandson/marji granddaughter and Kolami marge grandchild. Survey marjedɨ matches the complete same-language kinship formation with final ɨ versus i; preserve the gender-neutral grandchild meaning. The whole article includes this formation without requiring it to match the displayed head.'),
'burde':('d5415','DEDR5415 explicitly records Belari (LSB2.2) burde relative beside Tulu binne kinsman/relative/guest and other hospitality terms. Survey burde relatives matches the complete same-language form; the source plural English gloss does not establish a separate plural suffix or require a banquet gloss.')}
holds={
'haḷdi':(['d3999'],'Full DEDR3999 gives Tulu paḷi/pardi elder sister and Koraga hareddi/pareḍi, but the kinship sequence is introduced with a question mark. Bhat compares Belari haḷdi to Tulu paldi under p:h. This is a good local comparison; its placement in the old/aged article remains qualified by the dictionary itself. Hold pending resolution of that article-level uncertainty, not because the head must match.'),
'madiyeri':(['d4694'],'DEDR4694 includes Kannada madive/maduve and Tulu madůmè marriage. Belari madiyeri has unexplained -yeri relative to these comparanda. Full article read; do not reduce the complete source formation to an assumed mad- stem or fabricate a component link.'),
'tanne':(['d364'],'Tulu tāyi and other mother terms in DEDR364 do not explain tanne geminate n and short vowel. Form search finds unrelated cold/reflexive matches; those do not establish mother ancestry. Need a direct comparandum or a justified kinship formation.'),
'mēti':([],'Tulu wife search yields boḍedi and unrelated kinship terms; form search returns treading, fenugreek, chief and plough words with different senses or consonants. No supported whole-form wife etymology established; do not force a resemblance link.'),
'baggerɨ':([],'Tulu boy comparanda āṇů and paiyerů do not explain baggerɨ. Form search finds unrelated intention/bending/where words. No supported child-family assignment from these results; shared material with beggi girl needs independent evidence.'),
'beggi':([],'Form search finds Kannada fear verbs, while Tulu girl terms have different stems. No evidence yet derives Belari beggi from those words or explains it alongside baggerɨ boy. Preserve as a complete unresolved form.'),
'himbaḍɨ':(['d4715','d2410'],'Tulu son-in-law search points to marmāye and aḷiya. Full DEDR4715 does not explain Belari himbaḍɨ; 2410 is only a search lead, not a verified analysis. No reliable whole-form kinship route established.')}
for r in json.load(open(p/'inventory.json')):
 if r['ID'] in assigned or r['ID'] in seen:continue
 if r['Form'] in rules:
  d,e=rules[r['Form']];z['accepted'].append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=e+' Inheritance versus local transmission remains subject to review.',status='pending-review'))
 elif r['Form'] in holds:
  c,e=holds[r['Form']];z['held'].append(dict(record=r,reason=e,candidates=c))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');a=json.load(open(p/'primary-evidence.json'));ids={x['parent'] for x in z['accepted']};a.update({r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in ids and r['Status']!='duplicate-excluded'});(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');print(len(z['accepted']),len(z['held']))
