import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass266';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='8377a',citation='CDIAL[8377a]',evidence='The full CDIAL addendum 8377a explicitly gives Punjabi peṛ, Kotgarhi pēṛ, Jaunsari pēṛ and Hindi peṛ tree under pēḍa. This supports the selected e-vowel survey tree forms, preserving retroflex stop/flap notation and ordinary final vowel. Local phonetic history and cross-IA transmission remain qualified. The exact source-backed existing node 8377a is used; the separate curated same-spelling node f_rr7dv53h3a5pm is not silently merged or substituted.'),dict(parent='12060',citation='CDIAL[12060]',evidence='Full vīrudha gives Punjabi birwā plant, Nepali biruwā seedling, Bihari birwāi vegetable seedlings and Hindi birwā small plant/shrub. This supports Bagheli birbā as a regional birwā-family response. The survey broader tree gloss is preserved, with its local semantic extension and b/w variation explicitly qualified; no claim that every dictionary comparison means a mature tree. Cross-IA transmission remains uncertain.')]
sets=[{'peɖ','peɾ','peɾʌ','peḍ'},{'birbā'}];remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='tree':continue
 for i,fs in enumerate(sets):
  if r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare266.py').read_text());print('accepted',len(acc))
