import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass216';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='242',citation='CDIAL[242]',evidence='Full CDIAL adya explicitly gives Prakrit ajja, Punjabi/Lahnda ajj, Bengali/Oriya āj/āji, Gujarati āj and Marathi āj̈ today. The selected western adz and eastern Hajong adž/adži forms retain the affricate written with dz/dž; Dungra Bhili aje and Bhilali āje retain their final vowel. Kului adʒʰ/adʒ(ə) retain the source aspiration and optional schwa rather than being silently normalized. These simple today forms fit adya; the separate adyāpi 243 and adyāhnaḥ 244 entries have additional documented material and do not better explain them. Regional transmission and precise realization of source affricate symbols remain qualified; extra -ika/-iku/-ij forms are excluded.')]
sets=[{'adz','adž','adži','aje','āje','adʒʰ','adʒ(ə)'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='today':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
