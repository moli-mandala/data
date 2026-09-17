import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass174';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='6835',citation='CDIAL[6835]',evidence='Full CDIAL *dhūḍi/dhūli gives Phalura duṛi and Pashai duṛī/dūri, Punjabi dhūṛ/dhūl/dhor/dhūṛā, Oriya dhuḷi/dhūḷā, Gujarati dhūṛ/dhūḷ and Marathi dhūḷ. The addendum gives West Pahari dhvḷɔ, Jaunsari dhūḷ and Garhwali dhūḷū. Selected simple dust forms match these documented liquid/retroflex and aspiration variants; source vowels and initial retroflex notation are preserved as qualifications. Turner discusses competing deeper origins and possible influence of tuṣa; no new resolution of that issue or local IA transmission is asserted. Extended duṛli/dhudur and mixed responses are excluded.')]
sets=[set('dūṛī dṳr duli dhuḷu dɦoḷo duḍi dud dul dʊːɖə dʊːɭ dʰuːɖə ḍhur'.split())]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='dust':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
