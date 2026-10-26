"""Compare two independently extracted marker streams, without approving text."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_STREAM = HERE / 'tagged-blocks-v4.json'
SCOPE_STREAM = HERE.parent / 'scope/interlinear-native-block-inventory.jsonl'
root = json.loads(ROOT_STREAM.read_text())
scope = [json.loads(line) for line in SCOPE_STREAM.read_text().splitlines()]

def root_identity(row):
    return (row['section'], row['marker'], row['printed_number'], row['question_number'])

def scope_identity(row):
    return (row['section'], row['marker'], row['number_label'], row['question_label'])

assert len(root) == len(scope) == 637
assert [root_identity(r) for r in root] == [scope_identity(r) for r in scope]
assert len({r['physical_key'] for r in root}) == len(root)
assert all(not r['accepted_transcription'] for r in root + scope)
assert Counter(r['section'] for r in root if r['marker'] == 'p') == {
    'C.2': 34, 'C.3': 12, 'C.4': 39, 'C.5': 41}
script_g = [r for r in root if r['raw_marker'] == 'ɡ']
assert len(script_g) == 1
assert root_identity(script_g[0]) == ('C.3', 'g', '10', None)
questions = [r for r in root if r['marker'] == 'p' and r['question_number']]
assert len(questions) == 10 and all(r['section'] == 'C.5' for r in questions)
answers = [r for r in root if r['section'] == 'C.5' and r['marker'] == 'a']
assert len(answers) == 10
# Answer blocks contain source target text within backslashes. This guard
# prevents a future English-only classification from discarding the witness.
assert all(r['raw_text'].count('\\') >= 3 for r in answers)

missing = []
for section in ('C.2', 'C.3', 'C.4'):
    numbers = [r['printed_number'] for r in root
               if r['section'] == section and r['marker'] == 'n']
    for number in numbers:
        markers = Counter(r['marker'] for r in root if r['section'] == section
                          and r['printed_number'] == number
                          and r['question_number'] is None)
        for marker in ('g', 'f'):
            if not markers[marker]:
                missing.append({'section': section, 'number': number, 'marker': marker})
assert missing == [{'section': 'C.4', 'number': '3', 'marker': 'f'},
                   {'section': 'C.4', 'number': '20', 'marker': 'g'}]
report = {
    'status': 'Independent native-extraction structure reconciled; literal acceptance pending',
    'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT_STREAM, SCOPE_STREAM)},
    'matching_marker_events': 637,
    'phonetic_blocks': 126,
    'target_question_blocks': 10,
    'target_bearing_answer_blocks': 10,
    'script_g_marker': script_g[0]['physical_key'],
    'missing_printed_markers_requiring_original_review': missing,
    'not_proven': ['Literal glyph accuracy', 'Word-to-gloss alignment',
                  'Free-translation scope', 'Final emitted count',
                  'Language/profile/reference acceptance', 'Canonical installation'],
}
(HERE / 'scaffold-reconciliation.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
