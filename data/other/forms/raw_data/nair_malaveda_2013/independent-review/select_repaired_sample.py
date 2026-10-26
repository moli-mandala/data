"""Select new exact witness keys before original-backed revision review."""
import csv
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected = {
    'review-freeze.json': 'de21cde19b8a61715b1096e454b266e82fedd6e2fdecc2204164e1fd8d27b1a9',
    'proposed.csv': 'ee7fb5625bc4532815e64ae3755cd3ab76ec80c42d5b34fee25462af12e912c1',
    'proposal-audit.jsonl': '1e8b294a61b7bd05b648bd84edf22a367b559b7aa623d7dac124adb6235f1d3a',
}
for name, digest in expected.items():
    assert sha(PACKAGE / name) == digest, name
freeze = json.loads((PACKAGE / 'review-freeze.json').read_text())
for name, digest in freeze['files_sha256'].items():
    assert sha(PACKAGE / name) == digest, name
prior = json.loads((HERE / 'selection-20260927.json').read_text())
excluded = {e['witness_key'] for e in prior['selection']}
rows = {r[10]: r for r in csv.reader((PACKAGE / 'proposed.csv').open(newline=''))}
audit = [json.loads(s) for s in (PACKAGE / 'proposal-audit.jsonl').read_text().splitlines()]
rng = random.Random(20260927412)
selected = []
for page in range(40, 45):
    eligible = sorted((a for a in audit if a.get('entry_key') in rows and a['printed_page'] == page
                       and a['entry_key'] not in excluded), key=lambda a: a['entry_key'])
    for record in rng.sample(eligible, 4):
        row = rows[record['entry_key']]
        selected.append(dict(sample_index=len(selected) + 1, source_unit_key=record['source_unit_key'],
            witness_key=row[10], printed_page=page, actual_csv_row=row, actual_audit_record=record,
            original_image=f'pdf{page}-original.png', source_locator=row[7]))
assert len(selected) == len({e['witness_key'] for e in selected}) == 20
assert not excluded & {e['witness_key'] for e in selected}
output = dict(seed=20260927412, status='selected before fresh original-row comparisons',
    inputs_sha256={**expected, 'graph-proposal.json': sha(PACKAGE / 'graph-proposal.json'),
        'review-repair-continuity.json': sha(PACKAGE / 'review-repair-continuity.json'),
        'metadata/raw-title-evidence.json': sha(PACKAGE / 'metadata/raw-title-evidence.json'),
        'metadata/proposed-references.bib': sha(PACKAGE / 'metadata/proposed-references.bib')},
    source_pdf_sha256=freeze['source_pdf_sha256'], excluded_prior_witness_keys=sorted(excluded),
    method='Four actual emitted rows per original page40–44; twenty unique witness keys disjoint from prior twenty. Physical blocks may overlap because every p40/p44 lexical block was in prior review.',
    selection=selected)
(HERE / 'selection-repaired-20260927.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
for entry in selected:
    print(entry['sample_index'], entry['actual_csv_row'])
