"""Exact first-candidate continuity for three bound-typography tag repairs."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
old_path = HERE / 'pre-typography-revision/proposed.csv'
new_path = HERE / 'proposed.csv'
old = list(csv.reader(old_path.open(newline='')))
new = list(csv.reader(new_path.open(newline='')))
assert len(old) == len(new) == 135
changes = []
for before, after in zip(old, new):
    assert before[:14] == after[:14]
    if before[14] != after[14]:
        assert set(before[14].split()) - set(after[14].split()) == {'multiword-expression'}
        assert set(after[14].split()) <= set(before[14].split())
        changes.append(dict(entry_key=after[10], form=after[2], field='Tags', old=before[14], new=after[14],
            source_evidence='p56 morphology: space adjoining bound dash/hyphen in suffix or segmented form, not an independent lexical phrase. Literal printed spacing is retained.'))
assert {c['form'] for c in changes} == {'- il', 'avanu –e', 'en –atti'}
assert [r[2] for r in new if 'multiword-expression' in r[14].split()] == ['marattil ninnu']
(HERE / 'bound-typography-review.json').write_text(json.dumps(dict(status='Three source-supported Tags-only corrections; literal first reading unchanged',
    old_csv_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(), new_csv_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
    first_reading_sha256=hashlib.sha256((HERE/'physical-units-first-reading.jsonl').read_bytes()).hexdigest(),
    changes=changes, all_other_14_fields_unchanged=True, ordered_keys_unchanged=True,
    genuine_phrase='Malayalam marattil ninnu retains multiword-expression', canonical_changes=False), ensure_ascii=False, indent=2)+'\n')
