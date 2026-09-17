import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
assert not (P/'pass208-decisions.json').exists()
rules=[dict(parent='2574',citation='CDIAL[2574];CDIAL[3164]',evidence='Full CDIAL ka and kim belong to the interrogative paradigm but distinguish their reflexes: ka lists Romani kē/ke what, eastern ke and Kotgarhi kɛ, while kim lists Punjabi kī, Kalasha kīa and regional kyā. Bare survey ke/ke̤ in these NW and central/eastern lects does not alone select which historical inflection or remodeling underlies the response. A local paradigm or independently verified dictionary analysis is needed; uncertain cross-IA transmission alone is not the reason for holding.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
held=[dict(record=r,families=[0],reason=rules[0]['evidence'],passNumber=208) for r in json.loads((P/'inventory.json').read_text()) if r['ID'] in remaining and r['Form'] in {'ke','ke̤'} and r['Gloss'].lower().startswith('what')]
for suffix,obj in [('rules',rules),('decisions',dict(accepted=[],held=held))]:(P/('pass208-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
print('held',len(held))
