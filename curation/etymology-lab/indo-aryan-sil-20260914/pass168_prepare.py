import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass168';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='7695',citation='CDIAL[7695]',evidence='CDIAL *paṭalāli, the line of a thatch, specifically compares Gujarati paṛāḷ slope of a roof and paṛāḷī half of a sloping thatch/shed. The Bhil paḍāḷ/paḍāl roof forms match this regional long-ā-lateral pattern, with retroflex stop versus Gujarati flap and lateral notation retained. Turner marks the reconstruction/derivation as uncertain; this remains a qualified family link, with local IA transmission unresolved. Short-vowel paḍala could instead continue paṭala 7694 and is excluded. The similar 7693 film-over-eye family is not selected.')]
sets=[{'paḍāḷ','paḍāl','poḍāḷ'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='roof':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
held=[dict(record=r,families=[0],reason='Roof paḍala/paḍaḷa/poḍaḷ may continue paṭala 7694 or represent a shortened *paṭalāli 7695 family. Both full articles have relevant thatch senses; the local short-vowel form does not settle the morphological choice.',passNumber=168) for r in json.loads((P/'inventory.json').read_text()) if r['ID'] in remaining and r['Gloss']=='roof' and r['Form'] in {'paḍala','paḍaḷa','poḍaḷ'}]
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
