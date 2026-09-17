import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass287';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='6849',citation='CDIAL[6849,1]',evidence='Full dhūma gives northern dhū̃/dhūā̃, eastern dhũyā/dhõyā/dhuā̃, Gujarati dhūvɔ and dhumāṛɔ, plus Kachhi dhũāṛo and Jaunsari dhuwā̃ in the addendum. These support the selected simple and western retroflex-extended smoke forms. Preserve source vowels, gemination, aspiration and ḍ/ṛ spellings; local suffix histories and cross-IA transmission remain qualified. Both alternatives in the selected slash responses fit the family. Separate dhūmara/dhūmala/dhūmra articles were checked; they do not require moving the directly attested Gujarati-type extension from this family.'),dict(parent='6849-2',citation='CDIAL[6849,2]',evidence='Full section 2 dhūmikā or dhūmiyā gives Bshk dīmī smoke directly, supporting dīmīʔ/dīmī̃ʔ with glottalization and nasalization preserved. Do not reduce these to the unextended dhūma branch; cross-IA transmission is qualified.')]
sets=[('smoke',{'tū̃ā̃ / dū̃ā̃','dū̃ā̃ / dʰū̃','dʰūā̃ / dʰū̃','dʰūīyā̃','dʰūīā̃','dʰava','dhõa','dumːa','dõː','dʰãː','dũwə','dũwã','dʊwə','ḍhũwã','ḍhuwã','duvaḍu','dɦuvāḍu','dɦuvāṛu','duvāḍu','duvāḍo','dɦuvāḍā','dhuvāḍā','davaḍ','dhuvaḍ','duaḍa','dhuvaḍu','ḍhuvaḍu','thumāḍo','tumaḍi','tumaḍu','tumaḍo','tumaṛu'}),('smoke',{'dīmīʔ','dīmī̃ʔ'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare287.py').read_text());print('accepted',len(acc))
