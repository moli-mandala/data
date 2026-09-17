import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass198';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10071',citation='CDIAL[10071]',evidence='Full mārga article explicitly lists Bengali māgu/māug woman/wife (contemptuous) as derivatives following its discussion of the specialized hair-parting sense, and māgi loose woman. Hajong magu wife matches the neighboring Bengali lexical form; the survey does not mark contempt and that register is not imposed on it. The link follows Turner’s stated derivation, not an independently demonstrated road-to-wife semantic history. Inheritance versus regional IA borrowing remains unresolved. Full mātr̥grāma 10020 was inspected separately and documents Sinhala māgama woman/wife, not the exact Bengali māgu form.')]
sets=[{'magu'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='wife':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
