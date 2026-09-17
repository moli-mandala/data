"""Check only this acceptance's survival in an isolated complete build."""
import csv,json,sys,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
root=Path(sys.argv[1]); source=P.parents[2]
forms={r['ID']:r for r in csv.DictReader((root/'cldf/forms.csv').open())}
edges=list(csv.DictReader((root/'cldf/edges.csv').open()))
actual={(r['Child_ID'],r['Parent_ID'],r['Kind'],r['Rank'],r['Pos']) for r in edges}
rows=json.loads((P/'approved-assignments.json').read_text())
missing=[r for r in rows if (r['Form_ID'],r['Etymon_ID'],r['Kind'],r['Rank'],r['Pos']) not in actual]
assert not missing,missing[:10]
for m in json.loads((P/'before-approval-manifests.json').read_text()).values():
 for q in m['proposals']:
  for r in q['records']:
   current=forms[r['ID']]
   assert all(current[k]==r[k] for k in ['Language_ID','Form','Gloss']),r['ID']
   assert set(r['Source'].split(';'))<=set(current['Source'].split(';')),r['ID']
   assert set(r['Tags'].split())<=set(current['Tags'].split()),r['ID']
donors=json.loads((source/'data/other/params/raw_data/20260911-central-surveys-donors-audit.json').read_text())
for r in donors:
 current=forms[r['Persistent_ID']]
 assert all(current[k]==r[k] for k in ['Language_ID','Form','Gloss','Source']), (r,current)
 assert current['Status']=='entry'
assert len({r['ID'] for r in csv.DictReader((root/'cldf/forms.csv').open())})==len(forms)
oldids={r['ID'] for r in csv.DictReader((source/'cldf/forms.csv').open())}
refs={r['ID']:r for r in csv.DictReader((root/'cldf/references.csv').open())}
for k in ['platts1884','dehkhoda-vajehyab2026','centralbank-saral-kannada']:assert k in refs
report=dict(approvedRows=len(rows),approvedRecords=len({r['Form_ID'] for r in rows}),allApprovedEdgesPresent=True,targetIdentityMeaningSourcesAndDialectsPreserved=True,newDonorHeads=len(donors),donorFormsSourcesAndStatusesPreserved=True,missingPreviouslyCompiledIDs=sorted(oldids-forms.keys()),newCompiledIDs=len(forms.keys()-oldids),compiledForms=len(forms),compiledEdges=len(edges),referencesChecked=['platts1884','dehkhoda-vajehyab2026','centralbank-saral-kannada'])
(P/'compiled-acceptance-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
