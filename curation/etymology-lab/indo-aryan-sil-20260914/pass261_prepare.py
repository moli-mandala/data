import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass261';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='7918',citation='CDIAL[7918.1]',evidence='Full parṇa subsection 1 documents ordinary leaf forms Punjabi pannā, Nepali pānu, Marwari/Gujarati/Marathi pān and Gujarati pā̃dṛũ leaf with a consonantal extension. It explicitly gives reduced Kalasha pᵘŕə̃/pŕũ through a proposed puṇ stage. These support the selected nasal leaf forms and western consonant-extended family members. Source nasal place, retroflexion, aspiration, vowels and final inflection remain unchanged. The western ṭ/ḍ/ṭh variations are provisional regional-family assignments, not a newly reconstructed ancient suffix or settled sound law; local development and cross-IA transmission remain qualified. Ordinary leaf is distinguished from the separate pārṇa/betel-leaf discussion.')]
fs={'pụ̃','pọ̃','pānṭo','pānṭa','pānṭā','pānṭũ','pānaḍu','pānḍa','pāno','pāṇṭa','pāṇṭo','pānṭe','pānto','paṇḍho','pane','paŋ','paŋṭhe','paŋṭho','paŋe','paŋṭo','paŋṭa'}
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='leaf' and r['Form'] in fs:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare261.py').read_text());print('accepted',len(acc))
