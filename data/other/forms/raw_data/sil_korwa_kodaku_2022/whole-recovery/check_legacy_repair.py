"""Verify proposed legacy changes and parse them with the existing sound profile."""
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[5]
sys.path.insert(0, str(DATA))
os.chdir(DATA)
import make_cldf
import source_meta

old_path = DATA / 'data/other/forms/20260828-sil-korwa-kodaku.csv'
new_path = HERE / 'legacy-review/legacy-repaired-candidate.csv'
audit_path = HERE / 'legacy-review/legacy-metadata-repair.jsonl'
baseline = list(csv.reader(old_path.open()))
proposed = list(csv.reader(new_path.open()))
audit = [json.loads(line) for line in audit_path.read_text().splitlines()]
old = {r[10]: r for r in baseline}
new = {r[10]: r for r in proposed}
assert len(old) == len(audit) == 4458 and len(new) == 4459
assert set(new) - set(old) == {'silkorwakodaku2022:i183:K:f2:v2'}
assert not set(old) - set(new)
assert [r['entry_key'] for r in audit] == list(old)
changes = Counter()
for key, before in old.items():
    after = new[key]
    for index in (0, 1, 4, 7, 8, 9, 10, 11, 12, 13):
        assert before[index] == after[index], (key, index)
    for index in (2, 3, 5, 6, 14):
        changes[index] += before[index] != after[index]
    assert set(before[14].split()) <= set(after[14].split())
assert changes == {2: 9, 3: 203, 5: 9, 6: 4458, 14: 690}
assert sum(r[3] == 'small ant' for r in proposed) == 6
assert sum(r[3] == 'eldest brother' for r in proposed) == 2
assert new['silkorwakodaku2022:i183:K:f2'][2] == 'hʊwa'
assert new['silkorwakodaku2022:i183:K:f2:v2'][2] == 'hʊʔ ʌkʌrɪja'
assert not new['silkorwakodaku2022:i183:K:f2:v2'][11]
errors = io.StringIO()
parsed, stats = make_cldf.parse_file(str(new_path), errors, name='20260828-sil-korwa-kodaku')
assert not errors.getvalue(), errors.getvalue()
assert len(parsed) == 4459, len(parsed)
assert [r.entry_key for r in parsed] == [r[10] for r in proposed]
assert all(r.is_lone for r in parsed)
assert all('\ufffd' not in r.form for r in parsed)
report = {
    'status': 'Legacy change boundaries, keys and actual parser verified',
    'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (old_path, new_path, audit_path)},
    'existing_keys_preserved': 4458, 'proposed_rows': 4459,
    'new_keys': sorted(set(new) - set(old)),
    'parser_stats': stats, 'parser_errors': 0,
    'original_review': 'Root viewed PDF75/76/86 for all lexical repairs and PDF82/83/84/89/90 for metadata classes',
    'canonical_changes': False,
    'limitations': ['No full compiled identity/graph validation',
                    'No fresh rereading of all unchanged legacy lexical forms'],
}
(HERE / 'integration-proposal/legacy-repair-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
