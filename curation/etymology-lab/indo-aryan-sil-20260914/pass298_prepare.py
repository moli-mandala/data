import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass298';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5988',citation='CDIAL[5988]',evidence='Full traṭ explicitly gives Hindi taṛkā dawn and Old Marwari taṛako morning after crack/crackle verbs, comparing English crack of dawn. These support taḍke/tāḍke/taḍːka/taḍka/tʌɾʌke/ṭaṛaqā morning with source retroflex stops/flaps, gemination and vowels preserved. The addendum says most forms may instead be from √taṭ; keep that published uncertainty and regional transmission qualified.'),dict(parent='13067',citation='CDIAL[13067]',evidence='Full sakāla gives sakālam early in the morning, regional reduced forms and Oriya saaḷa/saaḷiā early. Western sakaye/sakai/sakay and sakaḍ fit this morning family provisionally, preserving source y/ḍ realizations of the expected liquid and qualifying the local sound history and cross-IA transmission. Do not link to unrelated sakala whole or śakala fragment.')]
sets=[('morning',{'taḍke','tāḍke','taḍːka','taḍka','tʌɾʌke','ṭaṛaqā'}),('morning',{'sakaye','sakai','sakay','sakaḍ'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare298.py').read_text());print('accepted',len(acc))
