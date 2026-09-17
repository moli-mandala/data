import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass233';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9237',citation='CDIAL[9237.1]',evidence='Full CDIAL biḍāla subsection 1 explicitly gives Maithili bilār/bilāri, Awadhi bilārī and Hindi bilār/bilāṛī cat; Prakrit feminine -liā forms are also documented. Dang Tharu bilariya fits this regional bilār- cat family with an ordinary feminine -iya ending. Survey rhotic and vowel notation is preserved and local IA transmission remains unresolved. The distinct short billa subsection is not used for this full bilār- form.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='cat':continue
 if r['Form']=='bilariya':
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 if r['Form']=='bilauri':held.append(dict(record=r,families=[0],passNumber=233,reason='Full CDIAL 9237.1 biḍāla has Nepali birālo, Kumauni birlāū and Hindi bilārī; subsection 2 billa separately has Lahnda biloṛī child and Hindi bilauṭā kitten. The bilauri cat family is plausible, but the au/r formation and exact subsection for the Dewas/Majhi forms are not resolved by these comparanda. This is a branch/morphology question, not merely cross-IA borrowing; seek a direct regional dictionary attestation before choosing a node.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare233.py').read_text());print('accepted',len(acc),'held',len(held))
