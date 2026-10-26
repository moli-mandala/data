"""Freeze proposed output and a stratified20-unit independent review packet."""
import csv
import hashlib
import json
import random
from pathlib import Path

from import_koya import AUDIT, HERE

P = HERE / 'whole-source-recovery'
rng = random.Random(2026092727)
additions = [json.loads(line) for line in (P / 'proposed-additions-audit.jsonl').read_text().splitlines()]
narratives = [a for a in additions if a.get('kind') in {'narrative-unit', 'test-question'}]
selected = []
for section in sorted({a['section'] for a in narratives}):
    selected.append(rng.choice([a for a in narratives if a['section'] == section]))
selected += rng.sample([a for a in narratives if a not in selected], 5)
for gloss in ['salt', 'six', 'different', 'village']:
    selected.append(rng.choice([a for a in additions if a.get('interpretation') and a['gloss'] == gloss]))
with AUDIT.open(newline='') as stream:
    words = [a for a in csv.DictReader(stream) if a['Status'] == 'installed']
for record in rng.sample(words, 4):
    selected.append({'source_unit_key': f"wordlist:{record['Site_Code']}:{record['Item']}",
                     'proposed_entry_keys': record['Entry_Keys'].split('|'),
                     'raw_wordlist_audit': record})
assert len(selected) == 20
outputs = {}
for kind in ['wordlist', 'narrative', 'comparative']:
    with (P / f'proposed-{kind}.csv').open(newline='') as stream:
        for row in csv.reader(stream):
            assert row[10] not in outputs
            outputs[row[10]] = {'file_kind': kind, 'row': row}
for record in selected:
    record['proposed_rows'] = [outputs[key] for key in record['proposed_entry_keys']]
edges = [a for a in additions if a.get('evidence_unit') or a.get('editorial_annotations')
         or a.get('expected_answer_notation') == 'parentheses']
for record in edges:
    record['proposed_rows'] = [outputs[key] for key in record['proposed_entry_keys']]
packet = {'seed': 2026092727, 'sample_units': selected, 'additional_edge_units': edges,
          'instructions': 'Compare actual proposed rows with original PDF and raw evidence, including forms, gloss scope, citations, dialects, notes, and exclusions. Report material errors; do not accept merely because tests pass.',
          'source_pdf': 'tmp/pdfs/koya/JLSR2021_029.pdf',
          'source_sha256': 'a6541e0d2397849ce7c36961b3849f3b2c1f1c267036cfa1a3f6025796e14e7d'}
(P / 'independent-review-selection.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2) + '\n')
files = [P / f'proposed-{k}.{ext}' for k in ['wordlist', 'narrative', 'comparative'] for ext in ['csv', 'yaml']]
files += [P / f'proposed-sil-koya-{k}.txt' for k in ['narrative', 'comparative']]
files += [P / name for name in ['proposed-additions-audit.jsonl', 'wordlist-note-relocation.jsonl',
                               'independent-review-selection.json', 'whole-draft-parser-validation.json']]
freeze = {'status': 'frozen output proposal awaiting independent review', 'rows': len(outputs),
          'sample_units': 20, 'additional_edge_units': len(edges),
          'sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
          'limits': 'Source-stage only. Bibliographic coverage and installation evidence still pending. No full CLDF/database build.'}
(P / 'whole-output-freeze-20260927.json').write_text(json.dumps(freeze, indent=2) + '\n')
print(json.dumps({k: v for k, v in freeze.items() if k != 'sha256'}))
