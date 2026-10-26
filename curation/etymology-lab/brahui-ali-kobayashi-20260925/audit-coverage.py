"""Reconcile all Brahui Ali-Kobayashi dictionary IDs and pending review manifests."""
import collections
import csv
import json
import sys
from pathlib import Path
P=Path(__file__).resolve().parent
R=P.parents[2]
sys.path.insert(0,str(R))
from etymology_assignments import read_assignments
source={r['ID']:r for r in csv.DictReader((R/'cldf/forms.csv').open())
        if r['Language_ID']=='Brahui' and r['Source'].startswith('ali-kobayashi2024[')}
assert len(source)==3368
linked={r['Child_ID'] for r in csv.DictReader((R/'cldf/edges.csv').open())
        if r['Rank']=='1' and r['Child_ID'] in source}
linked.update(r['Form_ID'] for r in read_assignments()
              if str(r.get('Rank'))=='1' and r['Form_ID'] in source)
assert len(linked)==2
scope=json.loads((P/'survey-inventory.json').read_text())
data=json.loads((P/'proposed-decisions.json').read_text())
unexamined=json.loads((P/'unexamined-inventory.json').read_text())
sets=[linked,{r['record']['ID'] for r in data['accepted']},
      {r['record']['ID'] for r in data['held']},{r['ID'] for r in unexamined}]
assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
assert set(source)==set.union(*sets)
assert {r['ID'] for r in scope}==set.union(*sets[1:])
assert [len(x) for x in sets]==[2,110,17,3239]
paths=json.loads((P/'review-manifest-paths.json').read_text())
manifest_ids=collections.Counter(record['ID'] for path in paths
    for proposal in json.loads(Path(path).read_text())['proposals']
    for record in proposal['records'])
assert manifest_ids==collections.Counter(sets[1])
rows=sum(len(proposal['assignments']) for path in paths
    for proposal in json.loads(Path(path).read_text())['proposals'])
assert rows==110
owners={id:'linked' for id in linked}
owners.update({id:'proposed' for id in sets[1]})
owners.update({id:'held' for id in sets[2]})
owners.update({id:'unexamined' for id in sets[3]})
report=dict(source='ali-kobayashi2024',language='Brahui',compiledSourceRecords=len(source),
    linkedRecords=len(linked),priorProposed=0,priorHeld=0,proposedRecords=len(sets[1]),
    heldRecords=len(sets[2]),unexaminedRecords=len(sets[3]),pendingBatches=len(paths),
    assignmentRows=rows,uniqueOwnership=True,manifestAgreement=True,recordOwnership=owners)
(P/'source-coverage-001.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print({k:report[k] for k in ('compiledSourceRecords','linkedRecords','proposedRecords','heldRecords','unexaminedRecords','pendingBatches')})
