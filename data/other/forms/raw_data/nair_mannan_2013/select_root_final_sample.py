"""Fresh physical-unit-disjoint sample; p49's four output units are exhausted."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(HERE / 'proposed.csv') == 'e14969d61a067d45143e4c842c9bc495d70f76a0fda76820f3d140a9e85bdbde'
previous = json.loads((HERE / 'root-original-review-pass1-20260927451.json').read_text())
old_units = {r['entry_key'].rsplit(':w', 1)[0] for r in previous['sample']}
rows = list(csv.reader((HERE / 'proposed.csv').open()))
sample, used = [], set(old_units)
for page in range(45, 49):
    pool = [r for r in rows if f':p{page}:' in r[10] and r[10].rsplit(':w', 1)[0] not in used]
    selected = []
    for row in sorted(pool, key=lambda r: hashlib.sha256(('20260927452:' + r[10]).encode()).hexdigest()):
        unit = row[10].rsplit(':w', 1)[0]
        if unit in used:
            continue
        used.add(unit)
        selected.append(row)
        if len(selected) == 5:
            break
    assert len(selected) == 5
    sample.extend({'page': page, 'entry_key': r[10], 'row': r} for r in selected)
assert len(sample) == 20
last_page = [r for r in rows if ':p49:' in r[10]]
assert len(last_page) == 4 and all(r[10].rsplit(':w', 1)[0] in old_units for r in last_page)
selection = {'status': 'selection only; independent manual reread pending', 'seed': '20260927452',
    'sampling': 'SHA256(seed:Entry_Key), five per p45–48 stratum, one output per physical unit, excluding every prior sampled physical unit',
    'p49_accounting': 'All four output units were sampled in first pass; no fresh p49 output unit exists. All four are targeted repair controls below, outside fresh20.',
    'csv_sha256': sha(HERE / 'proposed.csv'), 'audit_sha256': sha(HERE / 'proposal-audit.jsonl'),
    'sample': sample, 'p49_targeted_rows': last_page}
(HERE / 'root-final-sample-selection-20260927452.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
for item in sample:
    r = item['row']
    print(json.dumps([item['page'], r[10], r[2], r[3], r[14]], ensure_ascii=False))
