import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass289';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='6835',citation='CDIAL[6835]',evidence='Full dhūḍi/dhūli gives northern dhūṛ/dhūṛā and Nepali dhulo dust; its discussion explicitly notes dh~t variation in the wider dust vocabulary, possibly influenced by tuṣa. The selected tuṛ/tūṛ/tūrā̃ and ṭulo forms are provisional regional family matches, preserving unaspirated or retroflex t and nasalization rather than imposing a regular sound law. Both tūrā̃/tū̃r alternatives fit this qualified analysis. Cross-IA transmission remains unresolved.')]
sets=[('dust',{'tuṛ','tūṛ','tūrā̃','tūrā̃ / tū̃r','ṭulo'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare289.py').read_text());print('accepted',len(acc))
