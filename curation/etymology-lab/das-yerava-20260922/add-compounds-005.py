import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/das-yerava-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));seen={a['record']['ID'] for a in z['accepted']+z['held']}|{r['Form_ID'] for r in read_assignments()}
rules={
'Benge Neeru':(['d5517','d3690'],'DEDR5517 explicitly gives Kannada beṅge heat/fire, Tulu bēnkè heat and Kannada ben-nīr hot water. The source Benge Neeru is proposed as heat/hot plus water (3690 nīru), with both ordered constituents retained. The lexical first member is attested, but source-local inheritance versus transmission remains open.'),
'Kanja Neeru':(['d1458','d3690'],'DEDR1458 gives Kodagu kāy- with second stem kāñj- be hot/boil, alongside the southern heating family; 3690 supplies nīr/nīru water. Paniya Kanja Neeru is proposed as heated/boiled plus water. The -a form is compared with the nasal-palatal heated stem without assigning a precise grammatical tag from popular spelling.'),
'Bolle Kallu':(['d5496','d1374'],'DEDR5496(a) supplies the southern white/bright family, including Tulu boḷḷena whitish, boḷḷarè white and Kodagu boḷïtë white; 1374 gives Malayalam/Kannada kaḷḷu toddy. Paniya Bolle Kallu is tentatively analysed as white plus toddy. The source gloss itself only says Toddy; white is the proposed first-element interpretation, requiring review, not an added source translation.')}
for r in json.load(open(p/'survey-inventory.json')):
 if r['ID'] in seen or r['Form'] not in rules:continue
 ds,ev=rules[r['Form']];z['accepted'].append(dict(record=r,parent=ds[0],components=ds,kind='component',citation=';'.join(f'dedr[entry {d[1:]}]' for d in ds)+';'+r['Source'],evidence=ev+' DEDR nodes denote the full articles; no heading or source spelling is changed.',status='pending-review'))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
used={d for a in z['accepted'] for d in a.get('components',[a['parent']])};a={r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in used and r['Status']!='duplicate-excluded'};(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
