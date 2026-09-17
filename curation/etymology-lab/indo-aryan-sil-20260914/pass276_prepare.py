import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass276';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4918',citation='CDIAL[4918]',evidence='Full cōkṣa explicitly gives Gujarati cokhā rice, Sindhi cokho cleaned rice and Kachhi caukhā rice, alongside Prakrit cokkha/cukkha. These support the western cukhā/cukkā/cukā/tsoka/sukhā rice responses. The semantic development is documented rather than inferred only from clean. Source affricate/sibilant, aspiration, gemination, vowels and internal spacing remain intact; local phonetic history and cross-IA transmission are qualified.'),dict(parent='12415',citation='CDIAL[12415]',evidence='Full śāli explicitly gives Gujarati/Marathi sāḷ and regional sāl rice, with Sinhalese häl/äl also documented. These support western sāl/sāḷ and provisionally Noiri hal as a family member, retaining the uncertain local s-to-h history without claiming a Sinhalese loan. Source lateral quality and the broad rice gloss remain intact; unhusked/growing senses in the dictionary are not silently imposed on the source record. Cross-IA transmission is qualified.')]
sets=[{'tsu kha','cukhā','cukkā','cukā','tsoka','sukhā'},{'sāl','sāḷ','hal'}];remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='rice':continue
 for i,fs in enumerate(sets):
  if r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare276.py').read_text());print('accepted',len(acc))
