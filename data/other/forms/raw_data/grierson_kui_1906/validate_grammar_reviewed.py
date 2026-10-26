"""Bounded source-local coverage checks; does not build or install data."""
import hashlib
import json
import unicodedata
from pathlib import Path

P = Path(__file__).parent
load = lambda name: [json.loads(line) for line in (P / name).read_text().splitlines()]
before = load('grammar-first-reading.jsonl')
after = load('grammar-reviewed.jsonl')
old = {r['source_unit_key']: r for r in before}
new = {r['source_unit_key']: r for r in after}
assert len(old) == 431 and len(new) == len(after) == 432
assert set(old) <= set(new)
assert set(new) - set(old) == {'grierson1906kui:p462:noun-prose-ending:1'}
assert {k for k, r in old.items() if not r['form']} == {k for k, r in new.items() if not r['form']}
assert sum(not r['form'] for r in after) == 16
assert all(unicodedata.is_normalized('NFC', r['form']) for r in after)
assert new['grierson1906kui:p459:v-b-comparison:5']['form'] == 'ēañju'
assert new['grierson1906kui:p459:specimen-length:4']['form'] == 'ēāñju'
assert new['grierson1906kui:p463:causal-example:1']['uncertainty']
assert new['grierson1906kui:p462:principal-verbal-noun-c2:1']['uncertainty']
report = {'status': 'passed', 'physical_units': 432, 'preserved_first_reading_keys': 431,
          'source_empty_positions': 16,
          'sha256': hashlib.sha256((P / 'grammar-reviewed.jsonl').read_bytes()).hexdigest(),
          'limits': 'Coverage and selected adjudication assertions only; not independent source validation.'}
(P / 'grammar-reviewed-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
