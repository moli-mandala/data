import json
from pathlib import Path
P=Path(__file__).resolve().parent
allowed={8:{'belly/stomach','belly / stomach'},16:{'arm/hand','whole arm','arm/ hand'},26:{'you (inf.)','you (formal)','you (sing., informal)','you (informal)','you inf.','you (2nd sg, informal)','you (2nd sg, formal)','you (2 sg. informal)','you (2s, informal); you (2s, formal); you (2s, formal); you (2s, formal)','you (sing. informal)','you_(informal)','you (singular formal)'},40:{'new (thing)'},43:{'hot (water)'},94:{'mortar (for grain)'},116:{'we (you and us)','we (1st pl, us and you)','we (1st pl, us not you)','we (inc.)','we (exc.)','we (1st pl, inclusive)','we (1st pl, exclusive)','we (1p, inclusive); we (1p, exclusive)'},142:{'above/on top of','above / on top of'},143:{'short (object)','short','short (thing)'}}
cs=[]
for x in json.loads((P/'cross-gloss-candidates.json').read_text()):
 if x['gloss'] in allowed.get(x['family'],set()):cs.extend(dict(record=r,families=[x['family']]) for r in x['records'])
ids={x['record']['ID'] for x in cs};assert len(ids)==len(cs)
byid={r['ID']:r for r in json.loads((P/'inventory.json').read_text()) if r['ID'] in ids}
for x in cs:x['record']=byid[x['record']['ID']]
(P/'tenth-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
s=(P/'fourth_decide.py').read_text().replace('fourth','tenth').replace('passNumber=4','passNumber=10')
s=s.replace("elif re.search(r'uncertain", "elif not (l=='Rana' and 'Tharu-RNS-' in r['Tags']) and re.search(r'uncertain")
(P/'tenth_decide.py').write_text(s)
s=(P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','tenth-decisions.json').replace('sixth','tenth');(P/'tenth_save.py').write_text(s)
print('candidates',len(cs))
