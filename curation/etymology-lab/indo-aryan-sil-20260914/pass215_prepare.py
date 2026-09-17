import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass215';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4009-2',citation='CDIAL[4009]',evidence='Full CDIAL gati explicitly supplies an -(l)la extension with Hindi gayal/gail/galī path and Old Marwari gailo/galo road. The selected Bagheli geyil/geyl/geli/gelli/gəli and Kaithal gali path match this extended family; retain diphthong, schwa and gemination notation. The existing extension node is 4009-2, but the printed extension is unnumbered, so cite CDIAL 4009. These source path senses are distinguished from homonymous mortar elicitation responses. Local IA transmission remains unresolved.')]
sets=[{'geyil','geyl','geli','gelli','gəli','gali'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='path':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
