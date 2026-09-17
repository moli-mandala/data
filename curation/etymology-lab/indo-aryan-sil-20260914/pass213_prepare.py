import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass213';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='7766',components=['7766','6128'],citation='CDIAL[7766];CDIAL[6128];CDIAL[7753]',evidence='Full CDIAL padadaṇḍa explicitly compares the modern Hindi compound pagdaṇḍī footpath. Analyse these longer survey responses as pag/paga foot (padga 7766) plus daṇḍī/ḍaṇḍī ridge/path/stick (daṇḍa 6128), in that order. Turner gives Hindi pag foot/footstep and daṇḍa-family senses ridge of earth/path and Marathi footpath. Preserve epenthetic vowels, e/i variants, dental/retroflex spelling and g/k variation before the second member. CDIAL itself leaves single g and the vowel of padga reflexes unexplained; no new solution is asserted. Local IA borrowing is unresolved. These are ordered components of an attested modern expression, not wholesale assignment to the contracted padadaṇḍa family.')]
sets=[{'pakaḍaṇḍi','pagaḍaṇḍi','pakdaṇḍī','pagadaṇḍī','pagiḍaṇḍi','pegḍeṇḍi','pekḍeṇḍi','pagdanḍi','pagḍanḍi'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='path':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if 'components' in q else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
   if 'components' in q:x['components']=q['components']
   acc.append(x);break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print('accepted',len(acc))
