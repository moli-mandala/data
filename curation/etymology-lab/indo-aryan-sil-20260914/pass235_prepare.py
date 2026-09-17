import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass235';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11239',citation='CDIAL[11239]',evidence='Full CDIAL vatsa explicitly gives Gawri ēċī cow and derives it from vatsikā, citing NOGaw 27; it also gives West Pahari bachī cow and several heifer forms. The survey Gawri heʦī cow fits that directly documented lexical family. Initial h and affricate notation remain as elicited and are not silently regularized to the dictionary form. The source itself supplies the adult cow sense; regional transmission remains qualified.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='cow':continue
 if r['Language_ID']=='Gaw' and r['Form']=='heʦī':
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 if r['Form'] in {'vāsəḍi','vasaḍi','vasoro','vasaṛi'}:held.append(dict(record=r,families=[0],passNumber=235,reason='The broad vatsa calf family is plausible, but full 11239 vatsa, 11241 vatsatara and 11243 vatsarūpa offer distinct stems and extensions. In particular Gujarati vācharṛī calf under 11241 and Marathi vāsrū̃ under 11243 do not independently establish which history produces local vasaḍi/vasaṛi/vasoro cow. The adult cow sense occurs elsewhere in the broad family but is not sufficient to choose this regional stem and ending. Retain the exact cow gloss and seek a direct regional attestation; this is morphology/parent choice, not merely uncertain IA borrowing.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare235.py').read_text());print('accepted',len(acc),'held',len(held))
