import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass226';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9882',citation='CDIAL[9882];CDIAL[9883]',evidence='Full CDIAL markata 9882 is explicitly monkey, with Prakrit makkaḍa/makkaḍī and maṁkaḍa, Phalura mākaṛ, Oriya mākaṛa/māṅkaṛa, Gujarati mākṛũ/mākṛī and Marathi mākaḍ/makḍī. Selected simple survey monkey forms preserve dental/retroflex stop or rhotic notation, nasal and vowel differences, and feminine/ordinary final vowels. Adivasi Oriya makidi retains its less specific dental transcription rather than being silently normalized. The separate markaṭa 9883 has spider/insect meanings and is not the parent of these monkey records. The deeper possible Dravidian connection and local IA transmission remain unresolved; longer -iyu/-iya forms and substantially reduced maka/mako are excluded here.')]
sets=[{'mākaṛ','makṛi','maṇkəṛə','mākəḍi','makaḍi','makoḍ','makḍi','makoḍi','makiḍi','makoṛ','makor','makidi','məkəṛ','makoḍe','makoṛe'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='monkey':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
