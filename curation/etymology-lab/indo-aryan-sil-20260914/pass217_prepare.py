import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass217';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10539',citation='CDIAL[10539];CDIAL[10543]',evidence='Full CDIAL rakta gives Gujarati rātũ and Marathi rātā red and also includes Assamese rātul, Middle Bengali and Old Maithili rātula red under the same family. The selected Bhil/Bareli rātlo/ratalu/rataḷo-type red forms are linked as lateral-extended members of this family, preserving syncope, vowel endings and dental/retroflex t/l notation. This is not a claim of borrowing from the eastern languages or a reconstructed common lateral suffix; local formation and cross-IA transmission remain open. Full raktālu 10543 is explicitly red plus yam, and its Gujarati/Marathi reflexes mean yam/sweet potato, so that superficially similar plant compound is not selected for these colour responses.'),dict(parent='10539',citation='CDIAL[10539]',evidence='Noiri raṭu red matches the unextended rakta family, explicitly represented by western Gujarati rātũ and Marathi rātā in full CDIAL 10539. Preserve the source retroflex stop and short vowels as regional transcription/phonological qualifications. No extra lateral or nasal extension is posited; local IA transmission remains unresolved.')]
sets=[{'rātlo','ratalu','ratala','rātəlo','rātḷũ','rātḷā','ratəḷu','ratalo','rataḷo','raṭlo','raṭalu','raṭala','raṭalo'},{'raṭu'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='red':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
