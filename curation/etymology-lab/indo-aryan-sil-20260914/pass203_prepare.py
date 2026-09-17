import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass203';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='f_fofowqdh5shlk',citation='mewari;nirmaan2018mewari[p. 289, col. 2, entry 8]',evidence='Regional lexical-family membership only: the existing user-approved māṭī husband family is grounded in three Mewari survey attestations and the Mewari dictionary entry माटी maʈi husband, Kapasan area. Selected Rajasthanic/Bhil māṭī/maṭi/mati husband forms share this regional stem and sense; vowel length and dental/retroflex notation remain explicit qualifications. The family head is an attested representative, not a reconstructed ancestor. Ultimate ancestry and regional transmission remain unresolved. Homonymous soil/clay forms do not justify a historical link; extended maṭidoṭlo and other compounds remain excluded.')]
sets=[{'māṭī','maṭi','mati'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='husband':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
