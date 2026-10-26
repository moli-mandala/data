"""Prove exact scope of changes after independent Malaveda review."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE / 'revision2-pre-independent-repairs'
before = list(csv.reader((OLD / 'proposed.csv').open()))
after = list(csv.reader((HERE / 'proposed.csv').open()))
assert len(before) == len(after) == 223
assert [r[10] for r in before] == [r[10] for r in after]
changes = []
for a, b in zip(before, after):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            assert index in {11, 14}, (a[10], index)
            changes.append({'entry_key': a[10], 'field': 'Variant_Of_Key' if index == 11 else 'Tags', 'before': x, 'after': y})
assert sum(c['field'] == 'Tags' for c in changes) == 8
assert sum(c['field'] == 'Variant_Of_Key' for c in changes) == 12
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = {'status': 'only eight scoped tag repairs and twelve independently reviewed variant fields changed',
          'before_csv_sha256': sha(OLD / 'proposed.csv'), 'after_csv_sha256': sha(HERE / 'proposed.csv'),
          'before_audit_sha256': sha(OLD / 'proposal-audit.jsonl'), 'after_audit_sha256': sha(HERE / 'proposal-audit.jsonl'),
          'keys_and_order_unchanged': True, 'all_other_thirteen_fields_unchanged': True,
          'changes': changes, 'reference_change': 'Only explicit raw-title/editorial-normalization notes for two reported theses; no author/year/title display identity changes.',
          'canonical_changes': False}
(HERE / 'review-repair-continuity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'changed_tag_fields': 8, 'changed_variant_fields': 12, 'csv_sha256': report['after_csv_sha256'], 'audit_sha256': report['after_audit_sha256']}))
