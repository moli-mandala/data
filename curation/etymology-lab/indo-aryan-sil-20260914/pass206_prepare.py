import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass206';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9661',citation='CDIAL[9661]',evidence='Full CDIAL bhrātṛ explicitly gives Maiya žā, ky. ẓā and Chiliss ẓā brother, matching the selected younger-brother forms without additional morphology. The younger restriction is a survey sense, and the article also documents younger-brother usage in other NIA reflexes. Preserve postalveolar/retroflex notation; local IA transmission is not resolved.'),dict(parent='9661',citation='CDIAL[9661]',evidence='Full CDIAL bhrātṛ explicitly gives Awank bhirā brother alongside Lahnda bhrā and Punjabi bharā. The selected Awank pirā and Gojri pīrā younger brother are linked to this regional family with initial p versus bh/breathy articulation and vowel-length notation expressly qualified. This does not establish the local borrowing route; extra-ending piranden and mixed responses are excluded.')]
sets=[{'žā','ẓā'},{'pirā','pīrā'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='younger brother':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms and r['Language_ID'] in ({'Mai','Chil'} if i==0 else {'awan','Goj'}):
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
