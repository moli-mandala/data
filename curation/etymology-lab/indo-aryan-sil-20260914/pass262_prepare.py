import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass262';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='7969',citation='CDIAL[7969]',evidence='Full pallava explicitly gives Prakrit pallava and Palula palā leaf, Gujarati pālav/pāl foliage/shoots, Marathi pālā/pālẽ foliage and Kumaoni pālo vegetation. These support the western pālo/pālu/polā leaf responses as pallava-family members. The source singular leaf sense and vowel notation remain intact; local vowel history and cross-IA transmission remain qualified. This is the foliage/sprout head 7969, not the homonymous cloth-strip entry, and no pattra/parṇa substitution is made.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='leaf' and r['Form'] in {'pālo','pālu','polā'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare262.py').read_text());print('accepted',len(acc))
