import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass165';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11378',citation='CDIAL[11378.2]',evidence='CDIAL 11378 sense 2 explicitly supplies vardhanī broom, Prakrit vaḍḍhaṇī, Assamese bārni, Oriya baṛhaṇi/bāṛhni and eastern baṛhan/baṛhnī. The selected barhan/bherni/boreni forms match this nasal-final broom family, preserving aspiration, vowels and rhotic/retroflex notation with qualifications. The auspicious naming explanations are hypotheses reported by Turner; local IA transmission remains unresolved. The canonical parent is 11378, citing its specific sweeping sense.')]
sets=[{'barun','borḍʰən','bəhrni','baṛʰan','baṛən','bereni','bhaḍni','boreni','bedni','bhərni','bəhɹni'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='broom':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
