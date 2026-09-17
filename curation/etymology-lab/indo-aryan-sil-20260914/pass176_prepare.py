import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass176';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='4780',citation='CDIAL[4780]',evidence='Full CDIAL cikka mud/slime family explicitly supplies Lahnda cikkuṛ mud, Punjabi cikkaṛ, Hindi cīkaṛ mud/slime and Kotgarhi cikṛɔ mud in the addendum. Selected cikar/ciqaṛ/ciquṛ/cīqoṛ/cikor/cikoṛ responses retain this c-initial, medial-velar family, with q/k and vowels preserved as regional qualifications. This is distinct from k-initial *kicca 3153. The deeper Dravidian/Munda proposals and local IA transmission remain unresolved. The same-stem cikaṛ/cikoro alternatives are retained together, not split or discarded.'),
 dict(parent='4784',citation='CDIAL[4784.1]',evidence='Full CDIAL cikhalla subsection 1 supplies Hindi cihlā mud/ooze and cihel wet oozy land, alongside Prakrit cikhalla/cikhilla. Bagheli cehela/cehila mud fit the documented h-medial lateral form, preserving vowel variation and final ending. Subsection 2 cakhalla and subsection 3 cikkhalla have distinct comparanda and are not substituted. Local IA transmission remains unresolved.')]
sets=[{'cikar','ciqaṛ','ciquṛ','cīqoṛ','cikor','cikoṛ','cikaṛ / cikoro'},{'cehela','cehila'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='mud':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
