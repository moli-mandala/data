import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass192';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9926',components=['9926','6333'],citation='CDIAL[9926.1];CDIAL[6333]',evidence='Analyse the intact survey noon expression in source order as head/above plus day/sun. Full masta first branch gives mattha head/top, regional matho/mathē above and Gujarati māthũ head/summit; the mathe/māthe/mate element preserves vowel/aspiration differences. Full divasa gives day/sun reflexes, Prakrit divaha/diaha/dīha, Old Marwari dihāḍo and Gujarati dahāṛɔ. The dihi/ḍihi and dahaḍu/dāhaḍe/ḍahaḍu elements match those regional families, with retroflex notation and endings qualified. Day/sun overhead is the interpretation of the noon expression, not a newly reconstructed ancient compound. Turner allows crossing with ahar and Northwest spread of h-forms; transmission is unresolved. Preserve spacing/hyphens and save both lexical components.')]
sets=[{'mathe-dahaḍu','mātɦe dāhəṛe','māthe dāhaḍe','mathe dāhḍe','mathe-dihi','mate-dɦahaḍu','mathe-ḍihi','mathe-ḍahaḍu','mate-ḍhahaḍu'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='noon':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if 'components' in q else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
   if 'components' in q:x['components']=q['components']
   acc.append(x);break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print('accepted',len(acc))
