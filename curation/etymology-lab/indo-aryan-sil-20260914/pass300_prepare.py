import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass300';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5798',citation='CDIAL[5798,1]',evidence='Full tara section 1 explicitly records Bshk tar and Tor ta star. These directly support Bashkarik tar with source glottalization and Torwali tha with aspiration retained. The survey spelling does not establish a separate etymon; local phonetic development remains qualified.'),dict(parent='5798-4',citation='CDIAL[5798,4]',evidence='Full taraka section 4 records Gujarati taro and Marathi tara star. These support neighbouring Bhil taru forms, preserving the survey retroflex onset and final vowel. Cross-IA transmission and the precise local sound history remain qualified.'),dict(parent='4745',citation='CDIAL[4745]',evidence='Full candrana entry explicitly includes Marathi candni star alongside regional candani/candni moonlight and Multani candni light of moon or stars. This supports the selected candaini/candani/cadani star forms, including the lost nasal in cadani and expanded vocalism in candaini, as qualified regional members of this family. Preserve the source star gloss; neither a gloss error nor a specific borrowing route is asserted.'),dict(parent='6913',citation='CDIAL[6913]',evidence='Full naksatra entry gives star already in the Rigveda, with regional nakhattar/nakhat/nakhetar comparanda. Odia naksatra with the source vowel after t is a conservative or learned regional continuation of this lexical family; direct inheritance versus learned or cross-IA transmission is left qualified.')]
sets=[{'tāʔr','tʰā'},{'ṭaru'},{'candaīnī','candani','caḍani'},{'nakṣatṛa'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='star':continue
 for i,fs in enumerate(sets):
  if r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare300.py').read_text());print('accepted',len(acc))
