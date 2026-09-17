import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass210';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9051',citation='CDIAL[9051]',evidence='Full CDIAL phala explicitly places Lahnda phallī pod, Punjabi phalī seed-pod of any leguminous plant, Hindi phalī pod and Gujarati phaḷī together. The selected survey phalī/phaḷī/phelli groundnut forms identify a specific leguminous seed/pod with this lexical family. The groundnut restriction is attested by these surveys, not claimed for Sanskrit or for the CDIAL article. Bagheli phelli retains vowel/gemination notation. Repeated and lateral/non-lateral slash responses contain the same family and are preserved intact. Punjabi/Hindi or other cross-IA transmission is plausible, especially in the northwest, but no specific immediate donor route is asserted; the provisional reflex link records family membership.')]
sets=[{'pʰalī','pʰalī / pʰalī','pʰaḷī / pʰalī','phelli'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='groundnut':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))
