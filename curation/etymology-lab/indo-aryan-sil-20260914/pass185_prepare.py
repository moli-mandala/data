import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass185';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='4843',citation='CDIAL[4843]',evidence='Full CDIAL cīra includes Prakrit cīra/cīrī rag, Bihari cīr clothes in general, Maithili cīr clothes and Marathi cīr clothes/cirā strip of cloth. Selected Awankari/Goj ciṛa/ciṛā/ciṛe cloth match this rhotic cloth family, retaining the survey retroflex flap and vowel length as qualifications rather than silently normalizing them. Local IA transmission remains unresolved. Stop-final cida/ciḍā and extra-suffix forms are excluded.'),
 dict(parent='4910',citation='CDIAL[4910]',evidence='Full CDIAL cēla clothes/garment gives Domaaki čel dress/cloak, Pashai čilā cloth/dress and Oriya ceḷa cloth. Selected Goj cilo/ciḷo cloth match the lateral family with i-vowel, ending and lateral notation preserved as qualifications. The related cīra article is inspected separately; local IA transmission remains unresolved. Geminate ciḷːo and extra-suffix ciləṛā remain pending.')]
sets=[{'ciṛa','ciṛā','ciṛe'},{'cilo','ciḷo'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='cloth':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
