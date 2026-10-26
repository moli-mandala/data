"""Reconcile independent keyed metadata without overwriting reviewed literals."""
import ast
import hashlib
import json
from pathlib import Path

P = Path(__file__).parent
ROOT = P.parents[5]
REVIEW_SHA = '7fcd430cd2f6da86030bcffe61639cf41c39115834fb589306552009d94b99dd'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

review_path = P / 'independent-full-grammar-review-20260927.json'
assert sha(review_path) == REVIEW_SHA
review = json.loads(review_path.read_text())
source = P / 'grammar-reviewed.jsonl'
assert sha(source) == review['input_sha256']
rows = [json.loads(line) for line in source.read_text().splitlines()]
keys = [row['source_unit_key'] for row in rows]
assert len(keys) == len(set(keys)) == 432
assert set(keys) == set(review['coverage']['all_keys'])
by_key = {row['source_unit_key']: row for row in rows}
registry = ROOT / 'data/tags.py'
tree = ast.parse(registry.read_text())
allowed = set()
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'GRAMMATICAL_TAGS', 'GENDER_TAGS'} for t in node.targets):
        allowed.update(ast.literal_eval(node.value))
assert allowed
decisions = []
seen = set()
for proposal in review['metadata_proposals']:
    key = proposal['source_unit_key']
    assert key not in seen
    seen.add(key)
    assert proposal['field'] == 'tags'
    row = by_key[key]
    assert row['tags'] == proposal['old'], key
    assert set(proposal['proposed']) <= allowed, (key, set(proposal['proposed']) - allowed)
    row['tags'] = proposal['proposed']
    decisions.append(proposal)

key = 'grierson1906kui:p469:chinna-negative-participle:1'
row = by_key[key]
old = ['multiword-expression', 'verb', 'participle', 'neg']
assert row['tags'] == old
row['tags'] = old + ['conjunctive-participle']
decisions.append({'source_unit_key': key, 'field': 'tags', 'old': old,
                  'proposed': row['tags'],
                  'evidence': 'Root original p469 reread: explicitly labelled negative conjunctive participle; tmp/kui-grammar-original-leaf490.jpg.'})

ending = by_key['grierson1906kui:p462:noun-prose-ending:1']
assert ending['form'] == 'eñju'
ending['uncertainty'] += ' Independent review and root crop review find a possible vowel-length bar: eñju versus ēñju remains unresolved. Preserve the candidate literal without asserting short vowel.'
for row in rows:
    row['status'] = 'independent literal and metadata review reconciled; typed glyph and scope holds remain'
    assert set(row['tags']) <= allowed, (row['source_unit_key'], set(row['tags']) - allowed)
output = P / 'grammar-metadata-candidate.jsonl'
output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
original = [json.loads(line) for line in source.read_text().splitlines()]
assert [r['form'] for r in rows] == [r['form'] for r in original]
assert [r['source_unit_key'] for r in rows] == keys
report = {
    'status': 'source-local metadata candidate; not canonical installation',
    'input_sha256': sha(source), 'review_sha256': REVIEW_SHA,
    'registry_sha256': sha(registry), 'output_sha256': sha(output),
    'records': len(rows), 'blank_positions': sum(not r['form'] for r in rows),
    'metadata_decisions': decisions, 'literal_changes': 0,
    'root_review': 'All keyed proposals and evidence inspected. Retain source-explicit categories; no inferred etymological edges. Isolated noun-ending vowel length remains uncertain after source crop inspection.',
    'pending': ['grammar scope findings', 'typed glyph uncertainties', 'whole-source export and parser validation', 'canonical installation', 'full build and browser QA deferred by user'],
    'canonical_changes': False,
}
(P / 'grammar-metadata-status.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'records': len(rows), 'metadata_decisions': len(decisions), 'literal_changes': 0, 'output_sha256': sha(output)}))
