import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass306';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10304',citation='CDIAL[10304,1]',evidence='Full meghya section 1 explicitly records Pothwari meghlu and Punjabi meghla cloud. Bishnupriya meghala cloud matches this l-extended cloud family with a vowel breaking the cluster. Prefer the attested extended family to bare megha 10302; exact local development and cross-IA transmission are qualified, and a specific Punjabi donor is not asserted.')]
sets=[('cloud',{'meghala'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare306.py').read_text());print('accepted',len(acc))
