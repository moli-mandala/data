import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass173';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='11134',citation='CDIAL[11134]',evidence='Full CDIAL loṭha gives Prakrit loḍha rolling-pin, Awankari lōṛā roller, Kumaoni loṛo stone for grinding or pounding, Nepali lohoro stone rolling-pin, Bengali loṛā and Bihari loṛhā/loṛhī stone roller for spices. These directly support the selected survey pestle responses in the loṛ/loḍ/lohor family. Vowel, aspiration and rhotic/retroflex notation are retained. Grinding or rolling stone versus pounding pestle is a source implement distinction, not a changed gloss; local IA transmission remains unresolved.'),
 dict(parent='10223',citation='CDIAL[10223.1]',evidence='Full CDIAL musala section 1 supplies Oriya musaḷa, Gujarati musḷũ, Marathi musaḷ, Nepali musal and Hindi mūslī small pestle. Selected s-medial lateral forms fit this branch with vowels and endings preserved. Turner notes non-Aryan origin and explicitly marks some local IA loans; local transmission here remains unresolved. H-medial and liquid-lost responses are excluded because they require additional branch/phonological evidence.'),
 dict(parent='10223-2',citation='CDIAL[10223.2]',evidence='Full CDIAL musala section 2 is explicitly *muṣala or *muśala and gives West Pahari muśḷ/muśḷī pestle and Jaunsari mūśṛī, with the addendum again giving muśḷ. The survey ś-medial lateral pestle forms preserve the diagnostic sibilant of this branch; regional IA transmission is unresolved. This selects the combined canonical section 10223-2, not the separate database variant node *muśala.')]
sets=[set('loṛa lorʰa loro lorhī loḍa lorha loṛʰī lohora lora lorḍʰa loḍʰi lohoro ḷoḍī loḍi'.split()),set('mosḷiyā musol musil musoḷ'.split()),set('muśəḷu muślo muśəḷi'.split())]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='pestle':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
