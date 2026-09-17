import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass180';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5466-2',citation='CDIAL[5466.2]',evidence='Full CDIAL *ṭukk subsection 2 *ṭukka piece explicitly gives Gujarati ṭũk/ṭũkũ small/brief, alongside Punjabi ṭuk little and Hindi ṭuk a little. The selected simple ṭuk-/tuk- short responses match this regional adjectival use; nasalization, aspiration, gemination and dental/retroflex notation remain as transcribed and qualified. This uses the noun/adjective subsection 5466-2, not the cutting verb 5466. Turner calls the deeper connection with truṭ very doubtful; local IA transmission is also unresolved. Extra -l/-ḍ/-r forms are excluded pending evidence for their short sense.')]
sets=[set('ṭuko tuka ṭuka tuko tukõ toki tũku ṭukhu ṭukku'.split())]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='short':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
