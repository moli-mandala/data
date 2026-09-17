import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass170';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9996',citation='CDIAL[9996]',evidence='Full CDIAL *māḍa upper storey includes Prakrit māla upper part of a house, Gujarati māḷ loft/upper storey, māḷɔ loft/large house and specifically māḷiyũ loft; Marathi māḷā loft supplies a further regional lateral form. Survey māḷya/māḷiyā/maḷai/malu roof fit this upper-part-of-house family with retained vowel and ending variation. Roof versus loft is a qualified semantic identification, and local IA transmission remains unresolved. The remote Dravidian origin printed in the article is not an assertion of direct Dravidian borrowing by the survey languages. Search-linked māla 10088 instead means forest/garden and was rejected as evidence.')]
sets=[{'māḷya','māḷiyā','maḷai','malu'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='roof':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
