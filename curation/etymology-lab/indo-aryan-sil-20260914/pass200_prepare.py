import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass200';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9402',citation='CDIAL[9402]',evidence='Full bhartṛ husband article gives Pali bhattā/bhattāraṁ, Assamese bhatār, Bengali bhātār and Bihari bhatār husband. Hajong batar/batʰar husband match the eastern bhatār family; lack or shifted position of aspiration and vowel length are retained as survey qualifications. Local IA transmission remains unresolved. Extended bʰʌtaɾʌs and batevu are excluded pending morphology.'),dict(parent='9467',citation='CDIAL[9467]',evidence='Full *bhāriyāpa husband article explicitly groups Palula bharīu, Torwali be and Gawri heriou. The survey Palula bʰarīb/bʰāṛev, Torwali be/bve and Gawri hereo match those local comparanda, preserving b/v, flap and vowel variation as qualifications. This follows Turner’s grouping; the proposed compound bhāryā+pa is a reconstruction. His article records alternative *bharitṛ and Grierson’s vara comparison for Torwali, and questions whether Gawri h- arose from bh- in address. Those historical uncertainties are retained, with local IA transmission unresolved.')]
sets=[{'batar','batʰar'},{'bʰarīb','bʰāṛev','hereo','be','bve'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='husband':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
