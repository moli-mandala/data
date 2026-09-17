import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass169';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='3831',citation='CDIAL[3831]',evidence='Full CDIAL kharpara section 1 (r suffix) explicitly gives Punjabi khaprail tile, alongside Nepali khaprā, Bengali khaprā, Oriya khapara and Hindi khaprā tile. The survey khaprel/kheprel/khapparɛl roof forms match the specifically attested -rail extension with vowel contraction; roof is interpreted metonymically as tiled covering, without altering the elicited gloss. The article cautions that New IA has *kharpa/*khampa with varying suffixes, so a simple inherited kharpara development is not asserted. Local IA transmission is unresolved.')]
sets=[{'kʰapparɛl','kʰaprel','kheprel'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='roof':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
