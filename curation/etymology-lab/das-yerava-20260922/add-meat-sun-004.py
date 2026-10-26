import json,csv,gzip,sys
from pathlib import Path
root=Path.cwd();p=root/'data/curation/etymology-lab/das-yerava-20260922';sys.path.insert(0,str(root/'data'))
from etymology_assignments import read_assignments
z=json.load(open(p/'proposed-decisions.json'));seen={a['record']['ID'] for a in z['accepted']+z['held']}|{r['Form_ID'] for r in read_assignments()}
for r in json.load(open(p/'survey-inventory.json')):
 if r['ID'] in seen:continue
 d=None
 if r['Form'] in ['Rechi','Irechi'] and r['Gloss']=='Meat':d='d529';ev='DEDR529 explicitly gives Malayalam/Tamil iṟacci and Kodagu eraci flesh/meat. Paniya Irechi preserves the initial vowel and Ravula Rechi lacks it; both preserve the r-plus-affricate shape and exact meat referent. The vowel-less Ravula comparison remains a proposed correspondence, not a source emendation. The alternative Pengo etymology is not used.'
 if r['Form']=='Nera' and r['Gloss']=='Sun':d='d3774';ev='DEDR3774 explicitly gives Kodagu nēra sun/time and Malayalam nēram sun/day/light/time. Both source lects have Nera Sun, preserving the full solar noun rather than another near-homophonous water or standing stem.'
 if d:z['accepted'].append(dict(record=r,parent=d,kind='reflex',citation=f'dedr[entry {d[1:]}];'+r['Source'],evidence=ev+' Das documents multilingual contact; family membership is proposed while inherited versus locally transmitted history remains for review.',status='pending-review'))
 if r['Form']=='Curry':z['held'].append(dict(record=r,candidate='d1911 versus d1391',reason='Das explicitly glosses Curry cooked rice. DEDR1911 kūḻ/kūru family means boiled rice, but DEDR1391 kaṟi means vegetables/meat/curry rather than cooked rice. Popular English-like spelling cannot decide the source vowel and liquid; preserve the cooked-rice gloss and await same-lect evidence rather than assigning curry from spelling alone.'))
 if r['Form']=='Pure':z['held'].append(dict(record=r,candidate='d4328 versus d4565',reason='Das Pure firewood could be compared with the dry-wood family puḷḷe (DEDR4328) only with unexplained liquid/gemination differences. Kannada pore load includes bundles of firewood in a dictionary definition, but DEDR4565 itself is carry/load, not a direct firewood noun. Do not treat that gloss substring as lexical evidence.'))
(p/'proposed-decisions.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
used={a['parent'] for a in z['accepted']};a={r['Entry_ID']:r for r in csv.DictReader(gzip.open(root/'data/data/dedr/entry-texts-audit.csv.gz','rt')) if r['Entry_ID'] in used and r['Status']!='duplicate-excluded'};(p/'primary-evidence.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
