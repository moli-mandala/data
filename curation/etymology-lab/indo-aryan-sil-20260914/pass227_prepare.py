import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass227';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9696',components=['9696','8393'],citation='CDIAL[9696];CDIAL[8393]',evidence='Hajong masipoka/maśipoka fly is analysed as fly plus insect, in that order. Full CDIAL makṣā includes makṣikā, Bengali/Oriya māchi and other regional māśī/māsi fly forms; pōkka 8393 explicitly gives Bengali pokā and Oriya poka insect/worm. The source s/ś representation and vowel length remain unchanged and the first component is compared with the eastern fly family without asserting identical articulation to every cited reflex. Save both lexical components of the modern compound, with local IA transmission unresolved; no ancient compound node is invented.')]
sets=[{'masipoka','maśipoka'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='fly':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if 'components' in q else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
   if 'components' in q:x['components']=q['components']
   acc.append(x);break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print('accepted',len(acc))
