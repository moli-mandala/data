"""Verify the one visually confirmed missing-backslash boundary repair."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
old_path = HERE / 'tagged-blocks-v4.json'
new_path = HERE / 'tagged-blocks-v5.json'
old = json.loads(old_path.read_text())
new = json.loads(new_path.read_text())
assert len(old) == 637 and len(new) == 638
old_by_key = {r['physical_key']: r for r in old}
new_by_key = {r['physical_key']: r for r in new}
added = set(new_by_key) - set(old_by_key)
assert len(added) == 1 and not set(old_by_key) - set(new_by_key)
gloss = new_by_key[added.pop()]
assert (gloss['section'], gloss['printed_number'], gloss['marker']) == ('C.4', '20', 'g')
assert gloss['pdf_page'] == 102 and gloss['question_number'] is None
changed = [key for key in old_by_key if old_by_key[key] != new_by_key[key]]
assert len(changed) == 1
before = old_by_key[changed[0]]
after = new_by_key[changed[0]]
assert (after['section'], after['printed_number'], after['marker']) == ('C.4', '20', 'p')
assert before['source_lines'] == after['source_lines'] + gloss['source_lines']
assert 'and that time' not in after['raw_text']
assert 'and that time good beating also given' in gloss['raw_text']
report = {
    'status': 'One source-specific boundary repair verified; no literal source line lost',
    'original_review': 'Root and independent reviewer viewed PDF102 printed92 unit20: bare g precedes English gloss',
    'old_events': len(old), 'new_events': len(new),
    'unchanged_old_events': len(old) - len(changed),
    'corrected_phonetic_block': changed[0],
    'recovered_gloss_block': gloss['physical_key'],
    'source_lines_conserved': True,
    'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path, new_path)},
    'not_proven': 'Whole-text literal acceptance, word alignment and installation',
}
(HERE / 'bare-gloss-recovery-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
