import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass172';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='10990',citation='CDIAL[10990.1]',evidence='Full CDIAL laśuna section 1 gives Prakrit lasuṇa/lasaṇa/lhasuṇa, Gujarati lasaṇ, Marathi lasūṇ/lasaṇ, Hindi lasun/lahsun/lahsan and Assamese lahan. These document the sibilant and h-medial l-initial garlic family. Survey vowel variation, ś/ç notation and nasal place are preserved; ordinary endings are grouped, but m-final and n-initial forms are excluded. Local IA transmission remains unresolved, as also reflected by the article questioning Hindi influence on Punjabi.'),
 dict(parent='10990-2',citation='CDIAL[10990.2]',evidence='Full CDIAL section 2 *raśuna/rasuna/rasōna explicitly gives Bengali rasun, Oriya rasuṇa, Bihari rasun/rassun and Hindi rasun, distinct from the l-initial section. Survey r-initial garlic forms follow this branch, retaining vowel and rhotic/nasal notation; local IA transmission is unresolved.')]
sets=[set('loson losoṇ losan losun ləsune lahuṇ lohoṇ lohaṇo lohaṇ lahon lohuṇ lohəṇe lehoṇ lohon lasanə lesuṇ lasoṇ lasin lasiṇ laçiṇ laçuṇ ləśiṇ ləsoɳ lohoṇo ləṣun'.split()),set('rusun ṛəsuṇa rasuṇ rəsoɳ'.split())]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='garlic':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
