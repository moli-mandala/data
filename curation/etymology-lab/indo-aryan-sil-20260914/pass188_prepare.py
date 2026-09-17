import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass188';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='9883',citation='CDIAL[9883]',evidence='Full markaṭa² spider article gives Prakrit makkaḍa, Punjabi makkaṛ/makṛī, Nepali mākuro, Assamese makarā, Bengali mākaṛ, Oriya makaṛā and Hindi makṛā/makṛī. Selected simple survey spider forms preserve the corresponding makVr/makVḍ stem, with regional vowel quality, stop/flap notation and gender endings retained. Local IA transmission is unresolved. Longer -śa forms, extra morphology and multiword responses are excluded. The separately inspected 9884 markaṭaka article concerns grass and is not the parent.'),
 dict(parent='3535',citation='CDIAL[3535]',evidence='Full kōlika weaver article explicitly gives Prakrit kōlia weaver/spider and Marathi koḷī a sort of spider. Selected western koḷi/koḷio/kolyo and kuḷyo/kuḷyā/kuḷəyā spider forms match this lateral family with regional vowel raising and endings qualified. The article compares a deeper kōḍika/Munda connection; this link does not independently establish that remote origin or settle local IA transmission. Gujarati karoḷiyɔ is discussed separately with kaulāla and is excluded here.')]
sets=[{'mokura','makoṛa','mākəḍiyā','makaḍo','makaḍu','makor','makiḍa','mukuru'},{'kuḷyo','kuḷyā','kuḷəyā','koḷio','kolyo','koḷi'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='spider':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
