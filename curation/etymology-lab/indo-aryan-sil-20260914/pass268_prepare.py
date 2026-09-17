import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass268';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='8303',citation='CDIAL[8303.1]',evidence='Full puṣpa subsection 1 explicitly means flower and documents conservative puṣpa alongside later regional forms. Braj pūsp and Marathi pūṣpa are linked as conservative/learned flower-family forms, with final vowel loss, vowel length and learned or cross-IA transmission qualified. The separate puṣpya subsection 2 is not used.'),dict(parent='8306',citation='CDIAL[8306]',evidence='Full puṣya explicitly gives Torwali pašū, Palula piśīk, Gawri puṣa and Kalasha puṣík flower, distinguishing the ṣṣ development and discussing alternative historical analyses. These directly support the selected survey pūś-/piś- flower forms. Source vowels, sibilant notation and attested k extension remain intact; local transmission and ultimate stem history remain qualified. This is not an automatic puṣpa or phulla assignment.'),dict(parent='13846',citation='CDIAL[13846]',evidence='Full sphuṇṭati explicitly gives Bshk phuṇḍ full-blown/flower, Chil phundo flower, Gau phono and Shina Kohistani phuṇu. These support the selected Dardic nasal flower responses, preserving vowel nasalization, aspiration/fricative notation, retroflex stops/flaps and final inflection. The complete same-family slash response is retained. The nominal flower meaning is explicit; the ultimate sphuṭ/sphar root choice and local transmission remain qualified.')]
sets=[{'pūsp','pūṣpa'},{'pūśik','piśīk','pūśa','paśū','pośū'},{'fõṇṭ','pʰõṛ','pʰõḍ','pʰõṇṭ','pʰūṇḍo','pʰūṇḍo / pʰūnḍa','pʰū̃ṇḍū','pʰū̃ṇḍo'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='flower':continue
 for i,fs in enumerate(sets):
  if r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare268.py').read_text());print('accepted',len(acc))
