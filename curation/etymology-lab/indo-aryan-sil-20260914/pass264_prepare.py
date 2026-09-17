import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass264';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10250',citation='CDIAL[10250.1]',evidence='Full mūla subsection 1 explicitly gives Gujarati/Marathi mūḷ root, Marathi muḷī rootlet, Kumaoni mui root and Maldivian mū root, alongside lateral/rhotic regional variants. These support the selected western muḷ-/mul-/muy-/mō root responses and ordinary vowel/number forms. Source length, gemination, lateral quality and ending notation remain intact; exact local reductions and cross-IA transmission are qualified, without asserting that reduced forms were borrowed from Kumaoni or Maldivian. The separate Khowar mūḍa branch and unrelated mūrdhan comparison are not substituted.'),dict(parent='5086',citation='CDIAL[5086.1]',evidence='Full jaṭā subsection 1 gives Lahnda/Punjabi jaṛ root, Nepali jari/jaro root, Bihari jar root and Gujarati jaṛyā̃ root fibres, with Jaunsari jauṛ and Kotgarhi jɔṛh root in the addenda. These support the selected northern/eastern jaṛ/yaṛ/jaīr/joḍiya responses. Initial glide/affricate, rhotic/retroflex notation, diphthongs and local ending remain intact and qualified; regional transmission is uncertain. This uses the explicit root sense rather than inferring it only from matted hair, and does not invent a Dravidian or Munda donor.')]
sets=[{'mō','muḷe','muḷu','muyə','mulyā','muḷyā','mulːa','mule','muḷõ','muḷo','muḷːa','moː'},{'yaṛī','yaṛa','yaṛā','jaīr','jyaṛ','joḍiya'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='root':continue
 for i,fs in enumerate(sets):
  if r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare264.py').read_text());print('accepted',len(acc))
