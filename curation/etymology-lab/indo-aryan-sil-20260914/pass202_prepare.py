import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass202';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='f_dlo7xztxl5kpi',citation='mewari',evidence='Regional lexical-family membership only: the existing user-approved vīnd husband/bridegroom head is grounded in Mewari-survey vīnd husband, with the original phonemic ʋin̪d̪ retained in its donor audit. Selected neighboring Marwari/Dhundari bīnd/bīnda/bīndɨ/vīnda husband responses match this family with b/v variation and endings preserved. Ultimate ancestry and local transmission remain unresolved; this does not assert descent from Sanskrit bindu dot. The prior family audit reports the Hindi Śabdsāgar bridegroom sense as origin unknown, but its live page could not be fetched during this pass; no new claim is inferred from that failed request.')]
sets=[{'bīnd','bīnda','bīndɨ','vīnda'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='husband':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
