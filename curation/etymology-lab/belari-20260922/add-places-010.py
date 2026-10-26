import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/belari-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));assigned={r['Form_ID'] for r in read_assignments()};seen={x['record']['ID'] for x in z['accepted']+z['held']}
holds={
'sāro':([],'Tulu bridge search supplies only an unrelated supporting-pole expression; relaxed sāro search yields no matching article. Need a primary whole-form bridge comparison or documented borrowing route.'),
'oṭre':(['d946'],'DEDR946 supplies Tulu oṭṭe hole and wider breaking/hole formations, but Belari oṭre has an unexplained rhotic. Other form matches are unrelated verbs or poṭre tree hollow. Preserve the entire hole form pending sound or morphological evidence.'),
'aḍākina':([],'Tulu lid search supplies mucc- formations; relaxed aḍāk search has not explained complete aḍākina lid. Possible verbal formation requires independent morphological analysis, not an invented component division.'),
'kaḍvayi':([],'Tulu drum names do not explain kaḍvayi; the closest form-search hit is kadvāya cheek, a different meaning. No justified metaphor or compound analysis connects that hit to this drum word.'),
'bātre':(['d5515','d3821'],'DEDR5515 has veṟṟ-ilai and a questioned Tulu baccirè comparison competing with3821. Belari bātre is not directly explained by those forms; no full-form match from current search. Retain leaf gloss and do not create an unattested compound or donor.'),
'seyle':([],'Tulu firewood query gives unrelated fuel terms; relaxed seyle query finds no matching article. No supported whole-form firewood analysis yet; search absence is limited to these queries.'),
'hakkalɨ':(['d4423'],'Homepage Kannada hakkalu includes waste land without trees, but full DEDR4423 gives only questioned hakkalu gleanings within pick/gather. Belari plain cannot be assigned there solely from the search-added land sense. Need evidence linking the senses and resolving the dictionary uncertainty.')}
for r in json.load(open(p/'inventory.json')):
 if r['ID'] in assigned or r['ID'] in seen:continue
 if r['Form']=='teru':
  z['accepted'].append(dict(record=r,parent='d3422',kind='reflex',citation='dedr[entry 3422];'+r['Source'],evidence='DEDR3422 explicitly records Belari (LSB2.2) teru way, alongside Tamil teru street/highway/public road and Telugu teruvu way/road/path. Survey teru path or road matches the complete Belari form and compatible meaning. The questioned Kota comparison does not qualify the separately listed Belari attestation. Inheritance versus local transmission remains subject to review.',status='pending-review'))
 elif r['Form'] in holds:
  c,e=holds[r['Form']];z['held'].append(dict(record=r,reason=e,candidates=c))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n');a=json.load(open(p/'primary-evidence.json'));ids={x['parent'] for x in z['accepted']};a.update({r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in ids and r['Status']!='duplicate-excluded'});(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');print(len(z['accepted']),len(z['held']))
