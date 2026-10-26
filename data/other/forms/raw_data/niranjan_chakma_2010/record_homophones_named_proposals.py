"""Adjudicate remaining non-null proposals separately from definite corrections."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
prior = json.loads((ROOT / 'homophones-named-root-adjudication-20260927.json').read_text())
HELD = {
    ('homophones', 197, 1, 'definition_bn'),
    ('named-groups', 14, 2, 'target_form'),
    ('named-groups', 22, 3, 'target_form'),
}
decisions, inputs = [], {}
for section in ('homophones', 'named-groups'):
    path = ROOT / f'independent-{section}-complete-review-20260927.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == prior['input_sha256'][path.name]
    inputs[path.name] = digest
    report = json.loads(path.read_text())
    originals = {r['source_unit_key']: r for r in map(json.loads, (ROOT / f'{section}-first-reading.jsonl').read_text().splitlines())}
    for row in report['coverage']:
        for finding in row['findings']:
            if finding['status'] != 'proposed-correction':
                continue
            identity = section, row['printed_group'], row['group_cell'], finding['field']
            held = identity in HELD
            assert originals[row['source_unit_key']][finding['field']] == finding['old']
            decisions.append({'source_unit_key': row['source_unit_key'], 'field': finding['field'],
                              'old': finding['old'], 'accepted_value': None if held else finding['proposed'],
                              'proposed_value': finding['proposed'],
                              'root_decision': 'retain with typed transcription uncertainty' if held else 'accepted literal correction',
                              'pdf_page': row['pdf_page'], 'evidence': finding['evidence'],
                              'root_review': 'Compared original page renders; three insufficiently distinct letter/cluster readings remain typed holds, not silent emendations.'})
assert len(decisions) == 15
assert sum(d['accepted_value'] is not None for d in decisions) == 12
result = {'input_sha256': inputs, 'decisions': decisions,
          'accepted_fields': 12, 'held_fields': 3, 'canonical_changes': False,
          'policy': 'All earlier uncertainty records and literal group labels remain in the audit. Label correction does not establish dialect identity. This adjudication does not repeat or broaden the separate fresh 20-record audit.'}
(ROOT / 'homophones-named-proposals-root-adjudication-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'accepted_fields': 12, 'held_fields': 3}))
