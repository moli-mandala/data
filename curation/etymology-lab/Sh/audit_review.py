"""Validate pending review manifests without applying assignments."""
import csv, json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
R = Path(__file__).resolve().parent
D = R.parents[2]
sys.path.insert(0, str(D))
from assign_form_ids import validate_assignments
F = {r['ID']: r for r in csv.DictReader((D/'cldf/forms.csv').open())}
rows, saved, ids = [], [], set()
pq, rq, difficulty = Counter(), Counter(), Counter()
for path in sorted([*R.rglob('batch-*.json'), *(R.parent/'bro').glob('batch-*.json')]):
    m = json.loads(path.read_text())
    if m['status'] == 'saved':
        saved.extend(a for q in m['proposals'] for a in q['assignments'])
        continue
    queue = str(path.parent.relative_to(R.parent))
    for q in m['proposals']:
        target = set(q['formIds'])
        assert len(target) == len(q['formIds']), (path, q['number'], 'duplicate target')
        assert not target & ids, (path, q['number'], target & ids)
        assert target == {a['ID'] for a in q['attestations']} == {a['Form_ID'] for a in q['assignments']}
        assert all(i in F and not F[i].get('Redirect') for i in target)
        assert q['reviewStatus'] == 'pending'
        assert q['difficulty'] in {'straightforward', 'moderate', 'difficult'}
        assert q['evidence'].strip() and q['evidenceSource'].strip()
        ids.update(target)
        rows.extend(q['assignments'])
        pq[queue] += 1
        rq[queue] += len(target)
        difficulty[q['difficulty']] += 1
validate_assignments(list(F.values()), saved + rows)
overlay = list(csv.DictReader((D/'data/etymology-assignments.csv').open()))
accepted = {a['Form_ID'] for a in overlay if a['Status'] == 'accepted'}
assert not ids & accepted, ids & accepted
fields = ('Form_ID', 'Etymon_ID', 'Kind', 'Rank', 'Status', 'Pos')
key = lambda a: tuple(a.get(k, '') for k in fields)
overlay_keys = {key(a) for a in overlay}
assert all(key(a) in overlay_keys for a in saved)
audit = dict(checkedAt=datetime.now(timezone.utc).isoformat(),
    scope='Pending Shina and Brokskat review manifests, including verified saved-batch dependencies; no overlay write',
    pendingProposals=sum(pq.values()), pendingRecords=len(ids), assignmentRows=len(rows),
    proposalsByQueue=dict(pq), recordsByQueue=dict(rq), difficultyCounts=dict(difficulty),
    checks=['Unique pending target IDs across all review queues',
            'Attestation and assignment target sets agree',
            'Targets exist and are not redirects',
            'Pending review statuses and difficulty labels are valid; evidence is nonempty',
            'Combined proposed assignments validate with saved-batch dependencies',
            'Saved-batch dependency rows verified in current overlay',
            'No pending target has an accepted overlay row'],
    limitations=['Structural validation is not independent scholarly approval of each etymology.',
                 'Hard follow-ups and unexamined records remain outside these proposed assignments.'])
(R/'review-consistency-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(audit, ensure_ascii=False, indent=2))
