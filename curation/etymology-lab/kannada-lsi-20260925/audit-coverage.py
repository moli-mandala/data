"""Audit exact source-record ownership without modifying etymology sidecars."""
import collections
import csv
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sys.path.insert(0, str(ROOT))
from etymology_assignments import read_assignments

source = [r for r in csv.DictReader(open(ROOT / 'cldf/forms.csv'))
          if r['Language_ID'] == 'Kannada' and 'grierson-lsi1928' in r['Source']]
residual = json.loads((P / 'survey-inventory.json').read_text())
decisions = json.loads((P / 'proposed-decisions.json').read_text())
external = json.loads((P / 'externally-assigned-records.json').read_text())
source_by_id = {r['ID']: r for r in source}
assert len(source_by_id) == len(source) == 182
owners = {}
for name, records in [('proposed', [x['record'] for x in decisions['accepted']]),
                      ('held', [x['record'] for x in decisions['held']]),
                      ('external', external)]:
    for r in records:
        assert r['ID'] not in owners, (r['ID'], name)
        assert all(r[k] == source_by_id[r['ID']][k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags'])
        owners[r['ID']] = name
assert set(owners) == set(source_by_id)
assert {r['ID'] for r in residual} == {i for i, name in owners.items() if name != 'external'}
assert collections.Counter(r['researchStatus'] for r in external) == {
    'prior-pending-review': 49, 'accepted-sidecar': 1
}
accepted_ids = {r['Form_ID'] for r in read_assignments() if r['Status'] == 'accepted'}
assert {r['ID'] for r in external if r['researchStatus'] == 'accepted-sidecar'} <= accepted_ids
paths = json.loads((P / 'review-manifest-paths.json').read_text())
manifest_ids = []
for group in paths.values():
    for path in group:
        m = json.loads(Path(path).read_text())
        assert m['status'] == 'pending-review'
        for proposal in m['proposals']:
            manifest_ids += proposal['formIds']
assert collections.Counter(manifest_ids) == collections.Counter(x['record']['ID'] for x in decisions['accepted'])
out = {'sourceRecords': len(source), 'priorPendingReview': 49, 'priorAccepted': 1,
       'residual': len(residual), 'newPendingReview': len(decisions['accepted']),
       'researchHolds': len(decisions['held']), 'unexamined': 0,
       'exactRecordPreservation': True, 'uniqueOwnership': True,
       'reviewManifestAgreement': True}
(P / 'record-coverage-001.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
