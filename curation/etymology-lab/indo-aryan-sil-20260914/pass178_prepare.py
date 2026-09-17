import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass178';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='9069',citation='CDIAL[9069]',evidence='Full CDIAL *phāṭṭakka gate/door explicitly gives Punjabi, Nepali, Bihari and Hindi phāṭak gate, Assamese phaṭak gate and Gujarati phaṭkiyũ bārṇũ a single door shutter. Survey pʰaʈʌk door fits this gate/shutter family. Vowel length is retained as transcribed; Turner questions the deeper *prahaṭṭa derivation, which is not resolved here. Local IA transmission remains unresolved.'),
 dict(parent='6459',citation='CDIAL[6459]',evidence='Full CDIAL *duvāra explicitly gives Nepali duwār, Assamese duwār and Bengali duār/duor/dor door, beside Prakrit du(v)āra. The selected dor/dūvāīr/ḍuwar/ḍawar door responses match this syllabic duv-/du- family, preserving vowel and initial dental/retroflex notation as qualifications. They are distinguished from the dvāra 6663 bār family. Local IA transmission remains unresolved.')]
sets=[{'pʰaʈʌk'},{'dor','dūvāīr','ḍuwar','ḍawar'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='door':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
