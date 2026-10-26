"""Audit exact ownership of the current pending-review Tulu checkpoint."""
import csv
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
D = P.parents[2]
load = lambda name: json.loads((P / name).read_text())
inventory = load('survey-inventory.json')
decisions = load('proposed-decisions.json')
remaining = load('unexamined-inventory.json')
prior = json.loads((P.parent / 'dravidian-surveys-20260921/proposed-decisions.json').read_text())
prior_rows = [p['record'] for category in ('accepted', 'held') for p in prior[category]
              if p['record']['Language_ID'] == 'Tulu' and p['record']['Source'].startswith('grierson-lsi1928[')]
current = [p['record'] for category in ('accepted', 'held') for p in decisions[category]]
with (D / 'cldf/forms.csv').open(newline='') as stream:
    compiled = [r for r in csv.DictReader(stream)
                if r['Language_ID'] == 'Tulu' and r['Source'].startswith('grierson-lsi1928[')]
def keyed(rows):
    out = {r['ID']: r for r in rows}
    assert len(out) == len(rows)
    return out
all_rows, prior_map, current_map, remaining_map = map(keyed, (compiled, prior_rows, current, remaining))
assert len(all_rows) == 168 and len(prior_map) == 31 and len(current_map) == 137 and len(remaining_map) == 0
assert set(all_rows) == set(prior_map) | set(current_map) | set(remaining_map)
assert not (set(prior_map) & set(current_map) or set(prior_map) & set(remaining_map) or set(current_map) & set(remaining_map))
assert all(all_rows[k] == r for rows in (prior_map, current_map, remaining_map) for k, r in rows.items())
sys.path.insert(0, str(D))
from etymology_assignments import read_assignments
assigned = {r.get('Form_ID') for r in read_assignments() if r.get('Rank') == '1'}
assert not (set(current_map) & assigned)
report = dict(source='grierson-lsi1928', language='Tulu', totalCompiledSourceRecords=len(all_rows),
              priorPendingProposals=30, priorResearchHolds=1, newPendingProposals=len(decisions['accepted']),
              newProposedAssignmentRows=sum(len(x.get('components', [x['parent']])) for x in decisions['accepted']),
              newResearchHolds=len(decisions['held']), unexamined=len(remaining),
              exactRecordPreservation=True, uniquePassOwnership=True,
              alreadyRank1AssignedAmongPassRecords=0)
(P / 'record-coverage-002.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
