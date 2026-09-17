import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass278';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='12806',citation='CDIAL[12806]',evidence='Full ṣaṣṭikā gives Prakrit saṭṭhiya a sort of rice, Punjabi saṭṭhī coarse rice, Hindi sāṭhī/sā̃ṭhī rice ripening in sixty days and Marathi sāṭhī. Jaunsari saṭi fits this regional rice-name family provisionally, retaining the source unaspirated ṭ and vowel quantity without claiming a fully established local sound history. Preserve the broad survey gloss rice; the sixty-day ripening property is etymological evidence, not an added survey observation. Cross-IA transmission remains qualified.')]
sets=[{'saṭi'}];remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='rice':continue
 for i,fs in enumerate(sets):
  if r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare278.py').read_text());print('accepted',len(acc))
