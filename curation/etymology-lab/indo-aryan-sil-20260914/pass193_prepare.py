import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass193';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='644',components=['644','6333'],citation='CDIAL[644];CDIAL[6333];CDIAL[654]',evidence='Analyse the intact noon response as half plus day in source order. Full ardha² 644 explicitly cites Maiyã aṛ-dī noon beside regional aṛ/ar half; full divasa 6333 gives Maiyã dis and Shina dez day. The survey āṛdī/āṛdīs/āṛdez forms preserve these regional component shapes, with vowel length and final s/z variation stated. Full ardhadivasa 654 permits a locally formed compound rather than necessarily uninterrupted inheritance of the Sanskrit compound; ordered component links preserve that distinction. Local IA transmission and the age of the compound remain unresolved. The mixed āṛdīs / xārma response is excluded until its second alternative is independently analysed.')]
sets=[{'āṛdī','āṛdīs','āṛdez'}]
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
