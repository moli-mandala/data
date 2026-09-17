import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass236';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4883',citation='CDIAL[4883.1]',evidence='Full CDIAL cūḍa subsection 1 explicitly gives Bengali cul hair of head, Oriya cūḷa hair/lock and Assamese suli hair. Hajong tśul/tśuli hair fits this eastern family, preserving the affricate notation and final vowel. Ultimate history is qualified: Turner cites a Dravidian source and possible wider connection with the jaṭā/jūṭa hair group, not an established inherited Indo-European etymon. Regional IA transmission is unresolved.'),dict(parent='6582',citation='CDIAL[6582]',evidence='Full CDIAL dōla explicitly gives Prakrit ḍōla eye, Oriya doḷā/ḍoḷā pupil of eye and Marathi ḍoḷā eye. The selected Bhil-area ḍuḷu/ḍuḷo/ḍuḷā/ḍula/ḍuḷa/ḍolo responses fit the documented eye family, preserving vowel raising, dental/retroflex lateral notation and final inflection. The generic eye sense is directly documented, not inferred merely from swinging. Regional IA transmission remains unresolved; forms with v, glottal stop or a missing lateral are excluded for separate review.')]
sets=[{'tśul','tśuli'},{'ḍuḷu','ḍuḷo','ḍuḷā','ḍula','ḍuḷa','ḍolo'}];glosses=['hair','eye']
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,forms in enumerate(sets):
  if r['Gloss']==glosses[i] and r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare236.py').read_text());print('accepted',len(acc))
