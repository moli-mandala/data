import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass267';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9092',citation='CDIAL[9092]',evidence='Full phulla explicitly gives regional phul/phūl flower, Oriya phula, West Pahari fūl and Kotgarhi phulṛu. These support the selected simple phul/pul/fricative forms and western phulḍ- extensions through the documented rhotic extension. Source aspiration, fricatives, lateral quality, vowels and endings remain intact; the exact local extension and cross-IA transmission are qualified. This is the flower family, not phala fruit; nasal and unexplained k-initial forms remain separate.'),dict(parent='9051',citation='CDIAL[9051]',evidence='Full phala explicitly gives Oriya phaḷa, Gujarati/Marathi phaḷ, West Pahari phaḷ/phal and Kotgarhi phɔḷ fruit; Kumaoni phaw supplies a reduced-lateral comparison. These support selected fruit responses with preserved lateral/fricative notation and local glide/reduction variants. Source vowels and final inflection remain intact; exact local phonetic history and cross-IA transmission remain qualified. No phulla flower link or doubtful apple comparison is substituted.')]
sets=[{'pul','ɸʰula','phuləḍũ','phulḍũ','phulḍā','pulo','ɸol','ɸəl','ˈpʰuɭə','fʊːɭ','pʊɭ','pʊːɭ','phulḍa','ɸulo','ɸuḷ'},{'pol','pəḷə','ɸɔl','ɸaḷ','ɸoḷ','ɸoa','ɸaye','pəɭ','foːɭ','paːɭ','phole','phay','ɸola','ɸol'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,fs in enumerate(sets):
  if r['Gloss']==['flower','fruit'][i] and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare267.py').read_text());print('accepted',len(acc))
