"""Apply root-adjudicated literal readings in a separate, guarded derivative."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

first = HERE / 'p238-roman-first-reading.jsonl'
assert sha(first) == '5aa9e90379550c052fd281886f8d9e80f52d584e47e402049682ddbbfeb59586'
review_path = HERE / 'native-review/independent-roman-all-133-review.jsonl'
assert sha(review_path) == '1377fde229aa2cec3d02161b9c374a37e5f357fc7a1c1ad1f19c60e61aa3a370'
source_lines = rows(first)
atoms = [a for line in source_lines for a in line['target_atoms']]
reviews = rows(review_path)
assert len(atoms) == len(reviews) == 133
assert [a['source_unit_key'] for a in atoms] == [r['key'] for r in reviews]
for a, r in zip(atoms, reviews):
    assert (a['raw_form'], a['raw_gloss']) == (r['old_form'], r['old_gloss'])

accepted = {}
for r in reviews:
    if r['status'] == 'proposed-literal-correction':
        accepted[r['key']] = r['proposed_form']
    elif r['status'] == 'typed-source-glyph-hold':
        # Root inspected both native-pixel controls: each final h has an
        # independent lower dot, matching the printed underdot convention.
        assert r['old_form'] == 'kai-ṭarah' and r['alternative_form'] == 'kai-ṭaraḥ'
        accepted[r['key']] = r['alternative_form']
assert len(accepted) == 5
evidence = {}
for r in reviews:
    if r['key'] in accepted:
        p = ROOT / r['evidence']
        assert sha(p) == r['evidence_sha256']
        evidence[r['evidence']] = sha(p)
# Additional complete-name control corroborates the second and first Jai-.
full_name = HERE / 'native-review/followup-jai-full.png'
assert sha(full_name) == '83bae817a199fba71594a7a620ae870dfdb8526f4d3755773ff017a13507eb7c'

derived = copy.deepcopy(source_lines)
changed = []
for line in derived:
    line['first_reading_raw_target_line'] = line['raw_target_line']
    for a in line['target_atoms']:
        key = a['source_unit_key']
        if key in accepted:
            old = a['raw_form']
            a['first_reading_raw_form'] = old
            a['raw_form'] = accepted[key]
            a['literal_review_history'] = [{'old': old, 'new': accepted[key],
                'reason': 'Root corroborated the independently keyed proposal against the original native pixels.'}]
            changed.append(key)
        a['review_status'] = 'complete independent Roman/gloss review; root literal adjudication'
    line['raw_target_line'] = ' '.join(a['raw_form'] for a in line['target_atoms'])
    line['literal_boundary_policy'] = 'Printed spaces retain physical atoms; Jai-kaḷi-/kai is a separately recorded line-break continuation.'
flat = [a for line in derived for a in line['target_atoms']]
assert len(flat) == 133
for before, after in zip(atoms, flat):
    for field in before:
        if field not in {'raw_form', 'review_status'}:
            assert before[field] == after[field], (before['source_unit_key'], field)
assert set(changed) == set(accepted)

# Check proposed broad script spans as accounting evidence only. This does
# not certify the Native spellings or force word-to-word script equivalence.
native = rows(HERE / 'native-review/native-atoms-first-reading.jsonl')
spans = rows(HERE / 'native-review/native-roman-span-alignment-proposal.jsonl')
native_keys = [a['key'] for a in native]
assert len(native_keys) == len(set(native_keys)) == 194
assert [k for s in spans for k in s['native_keys']] == native_keys
assert [k for s in spans for k in s['roman_keys']] == [a['source_unit_key'] for a in atoms]

out = HERE / 'root-reviewed-revision'
out.mkdir(exist_ok=True)
def write(path, value, jsonl=False):
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in value) if jsonl else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == text, 'Refuse drift of an existing reviewed derivative'
    else:
        path.write_text(text)
write(out / 'p238-roman-reviewed.jsonl', derived, True)
write(out / 'p238-atomic-reviewed.jsonl', flat, True)
receipt = {'status': 'Roman/gloss literal review accepted; Native literal review remains unaccepted',
    'reviewer': '/root', 'first_reading_sha256': sha(first), 'independent_review_sha256': sha(review_path),
    'counts': {'physical_roman_lines': 15, 'roman_atoms': 133, 'literal_repairs': 5,
               'gloss_changes': 0, 'native_accounted_groups': 194, 'broad_script_spans': 8},
    'accepted_keyed_forms': accepted, 'evidence_sha256': evidence,
    'outputs': {p.name: sha(p) for p in out.glob('*reviewed.jsonl')},
    'remaining': ['Retain the ten original narrow transcription cautions until their individual scope is adjudicated.',
                  'Native spellings, typographic separator encoding and broad-script equivalences need independent literal review.',
                  'Joined lexical emission, whole expressions, documentary comparisons and final source integration remain separate gates.'],
    'canonical_changes': False, 'first_reading_rewritten': False}
write(out / 'root-literal-acceptance.json', receipt)
print(json.dumps({'status': 'PASS', 'repairs': 5, 'roman_atoms': 133, 'native_keys_accounted_only': 194,
                  'receipt_sha256': sha(out / 'root-literal-acceptance.json')}))
