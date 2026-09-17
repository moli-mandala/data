import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass234';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4147-3',citation='CDIAL[4147.3]',evidence='Full CDIAL gāva subsection 3 gāvī explicitly gives Prakrit gāī, Nepali/Bihari/Maithili gāi, and related feminine cow forms. The selected Tharu ɡaja/ɡʌjːa-type and Kochila gaⁱ/gaⁱye responses fit the regional gāi/gāy cow family. Survey j denotes the transcribed palatal glide; its length, nasalization and final vowel remain exactly as elicited, without asserting a separate ancient reconstruction for each ending. The extra gʌjijã response is excluded. Regional IA transmission and local ending formation remain qualified.'),dict(parent='4147-3',citation='CDIAL[4147.3]',evidence='Full CDIAL gāva subsection 3 gāvī explicitly gives Phalura ghāu cow and Awankari gā̃, plural gāī̃. The selected Phal gʰāo and Awankari gʰā survey responses are assigned to that specific branch; source aspiration, diphthong and nasal notation are preserved, rather than silently replacing them with the dictionary spelling. Local IA transmission remains unresolved.'),dict(parent='4147-2',citation='CDIAL[4147.2]',evidence='Full CDIAL gāva subsection 2 gāvā explicitly gives Kalasha gak cow, plural gāgan, with a tentative gāvakā comparison and alternative historical explanations through gām/gāḥ. Kalasha gakʰ cow fits this documented regional form; survey final aspiration remains intact and the historical alternatives remain qualified. The mixed gakʰ mẽṣ response is excluded.')]
sets=[{'ɡaja','ɡʌjːã','ɡʌ̃jːa','ɡʌjã','ɡɔja','ɡʌjːa','gaⁱye','gaⁱ'}, {'gʰāo','gʰā'}, {'gakʰ'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='cow':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare234.py').read_text());print('accepted',len(acc))
