import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass221';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11515',citation='CDIAL[11515]',evidence='Full CDIAL vānara gives both bānar and bā̃dar-type monkey reflexes, Oriya bāndara, Hindi bā̃dar/bā̃drā, Marwari bā̃dro and Gujarati vā̃dar/vā̃drɔ. The selected Bengali/Bishnupriya banor/banoɾ/bador, Bhatri bẽdṛa/bendṛa/be̩ndra and western/Tharu banḍar/baṇḍro/baṇḍorõ/bənḍra forms belong to this family. Preserve the eastern front-vowel notation, presence or absence of nasal spelling, dental/retroflex stops, schwa and final nasalization. These phonological qualifications and local IA transmission remain open; no new etymon or single borrowing route is asserted. The b(u)y-/buj- family and mixed monkey responses are excluded.')]
sets=[{'banor','banoɾ','bador','bẽdṛa','bendṛa','be̩ndra','banḍar','baṇḍro','baṇḍorõ','bənḍəra','bənḍra'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='monkey':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
