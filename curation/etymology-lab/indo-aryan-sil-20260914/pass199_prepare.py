import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass199';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10593-5',citation='CDIAL[10593.5]',evidence='Full raṇḍa subsection 5 explicitly gives Awankari ran wife, Lahnda rann woman/wife and Punjabi rann wife. The Awankari survey ran/ṛan woman/wife records match these regional forms, with retroflex-r notation preserved as a qualification. Other varieties in the article have widow or derogatory senses; none is imposed on the neutral survey gloss. The separately inspected rājñī queen article is not the source of these short ran forms. Local IA transmission remains unresolved.')]
sets=[{'ran','ṛan'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss'] not in {'woman','wife'}:continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
