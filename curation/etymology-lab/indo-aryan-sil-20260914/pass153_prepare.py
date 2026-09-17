import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass153';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5599-2',citation='CDIAL[5599.2]',evidence='CDIAL 5599 subsection 2 *dhera explicitly supplies Nepali dher/dherai much/many, distinct from the initial-retroflex head branch. Selected dental dher/dherai-type responses match this documented quantitative family, with vowel variation, deaspiration in dere and final inflection qualified. The local IA transmission route remains unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='many':continue
 if r['Form'] in {'dʰer','dʰjar','dʰerai̯','dʰerāi','dere','dherey','dheur'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 elif r['Form']=='ḍʰeṛ':held.append(dict(record=r,families=[],reason='CDIAL 5599 *ḍhera explicitly means much/many, but CDIAL 5598 *ḍheḍḍha explicitly yields Hindi ḍheṛ heap. Gojri ḍʰeṛ many requires evidence to decide between direct ḍhera with rhotic variation and the retroflex-r heap family with quantity extension. Do not resolve merely by the many gloss.',passNumber=153))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc),'held',len(held))
