"""Guarded No37 projection-only correction to the actual rich manual CSV schema."""
from pathlib import Path
import json, csv, hashlib, sys, io
from unittest.mock import patch

P = Path(__file__).resolve().parent
OLD = P / 'metadata-proposal-revision2'
NEW = P / 'projection-revision3'
NEW.mkdir(exist_ok=True)
ROOT = Path.cwd()
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD / 'candidate.csv') == 'fcd708e8edb9e164f28e8813bf69026bda262fa56b739fdde36814cd09862075'
assert sha(OLD / 'candidate-audit.jsonl') == 'df30840857b1e5899af75ecc436d91a29ed99904ace2624d768abbbef000af3f'
with (OLD / 'candidate.csv').open() as f:
    old = list(csv.reader(f))
audit = [json.loads(s) for s in (OLD / 'candidate-audit.jsonl').read_text().splitlines()]
assert len(old) == len(audit) == 390
projected = []
changes = []
for row, record in zip(old, audit):
    assert record['row'] == row and record['source_unit_key'] == row[10]
    new = list(row)
    new[1] = ''
    new[4] = row[1]
    assert all(new[i] == row[i] for i in range(15) if i not in {1, 4})
    projected.append(new)
    changes.append({'source_unit_key': row[10], 'old_row_guard': row,
                    'new_row': new, 'changed_indices': [i for i in range(15) if row[i] != new[i]],
                    'reason': 'Actual make_cldf.Row index1=Parameter_ID, index4=Native. Historical intended Native was misprojected into Parameter_ID; index2 remains the exact source Form/Original.'})
    record['row'] = new
    record['historical_projection'] = {'revision': 'metadata-proposal-revision2', 'row': row,
        'status': 'Rejected field projection; source literal fields themselves unchanged.'}
with (NEW / 'candidate.csv').open('w', newline='') as f:
    csv.writer(f).writerows(projected)
for name, records in [('candidate-audit.jsonl', audit), ('all-390-projection-changes.jsonl', changes)]:
    (NEW / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))

# Literal source-local parser fixture, not a canonical registration or proposed
# installation metadata replacement. The real parser and real SourceMeta class
# are used with only the load() provider patched to this explicit fixture.
yaml = NEW / 'candidate.yaml'
yaml.write_text('''file: candidate.csv
defaults:
  transcription:
    convert: false
    preserve_hyphens: true
  identity:
    dedupe_by_entry_key: true
sources:
  grierson1908ahirwati:
    transcription:
      convert: false
      preserve_hyphens: true
    forms:
      split_alternates: false
    gloss:
      grammar_tags: false
''')
sys.path.insert(0, str(ROOT / 'data'))
import os
os.chdir(ROOT / 'data')
import make_cldf, source_meta
meta = source_meta.SourceMeta(paths=[yaml])
errors = io.StringIO()
with patch.object(source_meta, 'load', return_value=meta):
    parsed, _ = make_cldf.parse_file(str(NEW / 'candidate.csv'), errors, name='candidate', file_num=999)
assert not errors.getvalue(), errors.getvalue()
assert len(parsed) == 390
for raw, actual in zip(projected, parsed):
    assert actual.param == ''
    assert (actual.old_form, actual.native, actual.ipa, actual.entry_key) == (raw[2], raw[4], raw[5], raw[10])
    assert actual.gloss == raw[3], (raw[10], actual.gloss, raw[3])
    assert actual.notes == raw[6]
    assert actual.source == raw[7]
    assert (actual.variant_of_key, actual.borrowed_from_key, actual.derivation_parent_keys) == tuple(raw[11:14])
    assert set(actual.tags.split()) == set(raw[14].split())
for k in ['lsi9.2:p234:aligned-line:4:word:3', 'lsi9.2:p234:aligned-line:4:word:4']:
    assert next(r for r in projected if r[10] == k)[4] == ''
report = {'status': 'passed projection-only correction and actual production parser check; source-local proposal only',
    'counts': {'rows': 390, 'protected_other_indices_per_row': 13, 'keys_changed': 0,
               'metadata_tag_changes': 0, 'actual_parser_rows': len(parsed)},
    'schema': {'index1': 'Parameter_ID blank', 'index2': 'source Form input / parsed Original',
               'index4': 'Native; exact previously intended Native', 'index5': 'Phonemic blank'},
    'limits': 'Source-local convert:false parser fixture verifies actual schema without canonical registration. Display Form follows the generic literal route; exact terminal punctuation is certified in Original, not asserted invariant under generic display normalization. Pending independent metadata decisions are not mixed into this revision.',
    'inputs': {str(f.relative_to(P)): sha(f) for f in [OLD / 'candidate.csv', OLD / 'candidate-audit.jsonl']},
    'outputs': {f.name: sha(f) for f in NEW.iterdir() if f.is_file()},
    'parser_source_sha256': sha(ROOT / 'data/make_cldf.py'),
    'script_sha256': sha(Path(__file__))}
(NEW / 'projection-only-continuity-parser-review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False))
