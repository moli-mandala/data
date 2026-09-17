import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass299';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4661',citation='CDIAL[4661,2]',evidence='Full candra section 2 gives Bshk čən, Chilisso čan, Lahnda cann/can, regional cand/cā̃d/cān and Assamese sā̃d/sān, alongside Gujarati cā̃do. These support the selected affricate/sibilant moon forms with source velar versus dental nasal spelling, retroflexion, vowel nasalization and ordinary endings preserved. Both can/cand alternatives fit. Local sound history and cross-IA transmission are qualified; extra syllables and moonlight-derived responses are excluded.'),dict(parent='13574-2',citation='CDIAL[13574,2]',evidence='Full sūrya section 2 gives Prakrit sujja and Lahnda suj sun. These support sujo, alongside conservative regional śurjo/ʃuɹdʒo/suɹya with the r-containing cluster retained or adapted. Preserve source consonants and vowels; conservative or learned/cross-IA transmission is qualified. Do not collapse the distinct sūriya and sūrī branches into this one.')]
sets=[('moon',{'tśan','caŋḍ','saṇḍ','tsaŋḍ','tsand','sãḍ','can / cand','sandra','tsʰan','cana','tsan','caṇḍa','saṇṭ','canḍe','sãḍe','tsaŋ'}),('sun',{'sujo','śurjo','ʃuɹdʒo','suɹya'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare299.py').read_text());print('accepted',len(acc))
