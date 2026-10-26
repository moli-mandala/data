"""Independently prove Roman emission granularity and revision continuity."""
import csv, json, hashlib, collections
from pathlib import Path

P = Path(__file__).resolve().parent
S = P.parent
OLD = P / 'metadata-proposal'
NEW = P / 'metadata-proposal-revision2'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def records(p):
    return [json.loads(s) for s in p.read_text().splitlines()]

def csv_rows(p):
    with p.open() as f:
        return {r[10]: r for r in csv.reader(f)}

assert sha(OLD / 'candidate.csv') == '97380267c11c0cc28def90ebc13e326a9b86f922a38ca77c1040ae01d978e401'
assert sha(OLD / 'candidate-audit.jsonl') == '593422efc8c3cb2cc782496a4a19d342673fb9afcdab15c42ab0cc37ebb12ce3'
old = csv_rows(OLD / 'candidate.csv')
new = csv_rows(NEW / 'candidate.csv')
head = 'lsi9.2:p234:aligned-line:4:word:3'
added = 'lsi9.2:p234:aligned-line:4:word:4'
assert set(new) - set(old) == {added}
assert not set(old) - set(new)
changed = {k for k in old if old[k] != new[k]}
assert changed == {head}
assert old[head][2:4] == ['bā̃ṭ diyō.', 'having-divided it-was-given.']
assert new[head][1:4] == ['', 'bā̃ṭ', 'having-divided']
assert new[added][1:4] == ['', 'diyō.', 'it-was-given.']
assert {'verb', 'pret', 'pass'} <= set(new[added][14].split())
assert all(new[k] == old[k] for k in old if k.endswith(':expression'))
roman = records(S / 'roman-atoms-first-reading.jsonl')
native = records(S / 'native-atoms-first-reading.jsonl')
joins = records(NEW / 'reviewed-printer-continuations.jsonl')
assert len(joins) == 4
tails = {j['physical_keys'][1] for j in joins}
expected = {r['key'] for r in roman} - tails
atomic = {k for k in new if not k.endswith(':expression')}
assert atomic == expected and len(atomic) == 363
aliases = records(NEW / 'physical-key-aliases.jsonl')
assert collections.Counter(a['physical_key'] for a in aliases) == collections.Counter(r['key'] for r in roman + native)
groups = {g['source_group_key']: g for g in records(NEW / 'shared-native-groups.jsonl')}
assert len(groups) == 1
for a in aliases:
    if a['emission_key'] is None:
        assert a['source_group_key'] in groups
        assert a['physical_key'] in groups[a['source_group_key']]['native_physical_keys']
    else:
        assert a['emission_key'] in atomic
report = {
    'status': 'passed source-local granularity refinement; not canonical installation',
    'counts': {'Roman_physical_atoms': 367, 'reviewed_printer_joins': 4,
               'Roman_lexical_emissions': 363, 'whole_expressions': 27,
               'candidate_rows': 390, 'Native_physical_atoms': 365, 'physical_aliases': 732},
    'continuity': {'historical_rows': 389, 'removed_keys': [], 'added_keys': [added],
                   'changed_common_keys': [head], 'unchanged_common_rows': 388,
                   'whole_expression_rows_unchanged': 27},
    'script_segmentation_decisions': [
        {'Roman': 'bā̃ṭ + diyō.', 'Roman_emissions': 2, 'Native': 'बाँट-दियो।',
         'policy': 'Both lexical Native fields blank; full one-word Native span retained in shared group audit.'},
        {'Roman': '‘achchhā- + taĩ-achchhā', 'Roman_emissions': 1,
         'policy': 'Independent fourth printer continuation; full two-token Native span retained.'},
        {'Roman': 'jamā-kar-kē', 'Roman_emissions': 1,
         'policy': 'One printed Roman word; full two-token Native span retained.'}],
    'hashes': {str(f.relative_to(P)): sha(f) for f in [
        OLD / 'candidate.csv', OLD / 'candidate-audit.jsonl', NEW / 'candidate.csv',
        NEW / 'candidate-audit.jsonl', NEW / 'status.json',
        P / 'fourth-printer-continuation-review.json', Path(__file__)]},
    'limits': 'Metadata remains proposed for parent review; no canonical/shared changes, phonemic inference, or new graph links.'}
(P / 'revision2-granularity-continuity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False))
