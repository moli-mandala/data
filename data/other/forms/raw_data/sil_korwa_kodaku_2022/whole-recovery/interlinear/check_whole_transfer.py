"""Independently reconcile all whole expressions against repaired marker evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
events_path = HERE / 'tagged-blocks-v5.json'
rows_path = HERE.parent / 'interlinear-review/whole-expression-proposal.jsonl'
events = json.loads(events_path.read_text())
rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
keyed = {r['source_unit_key']: r for r in rows}
assert len(keyed) == len(rows) == 117

def content(event):
    raw = event['raw_text'].strip()
    pattern = r'^g\s+' if event.get('marker_recovery') else r'^\\[npgɡfqa]\s*'
    return ' '.join(re.sub(pattern, '', raw, count=1).split())

used_p = Counter()
used_answers = Counter()
for row in rows:
    section = row['section']
    if row['role'] == 'whole-narrative-expression':
        unit = row['printed_number']
        selected = [e for e in events if e['section'] == section
                    and e['question_number'] is None
                    and (e['printed_number'].split('.')[0] if section == 'C.5'
                         else e['printed_number']) == unit]
        ps = [e for e in selected if e['marker'] == 'p']
        fs = [e for e in selected if e['marker'] == 'f']
        gs = [e for e in selected if e['marker'] == 'g']
        assert len(fs) <= 1
        assert row['form'] == ' '.join(content(e) for e in ps), row['source_unit_key']
        assert row['gloss'] == (content(fs[0]) if fs else ''), row['source_unit_key']
        assert row['raw_interlinear_glosses'] == [content(e) for e in gs], row['source_unit_key']
        assert row['physical_p_keys'] == [e['physical_key'] for e in ps]
        used_p.update(row['physical_p_keys'])
    else:
        q = row['question_number']
        selected = [e for e in events if e['section'] == section and e['question_number'] == q]
        if row['role'] == 'whole-target-question':
            ps = [e for e in selected if e['marker'] == 'p']
            qs = [e for e in selected if e['marker'] == 'q']
            assert len(ps) == len(qs) == 1
            assert row['form'] == content(ps[0]) and row['gloss'] == content(qs[0])
            assert row['physical_p_keys'] == [ps[0]['physical_key']]
            used_p.update(row['physical_p_keys'])
        else:
            assert row['role'] == 'source-target-answer'
            answers = [e for e in selected if e['marker'] == 'a']
            assert len(answers) == 1
            match = re.fullmatch(r'(.*?)\\([^\\]+)\\\s*', content(answers[0]))
            assert match
            assert row['form'] == match[2].strip() and row['gloss'] == match[1].strip()
            assert row['physical_answer_key'] == answers[0]['physical_key']
            used_answers.update([row['physical_answer_key']])
assert used_p == Counter(e['physical_key'] for e in events if e['marker'] == 'p')
assert used_answers == Counter(e['physical_key'] for e in events
                               if e['section'] == 'C.5' and e['marker'] == 'a')
report = {
    'status': '117 whole-expression/answer rows agree with repaired native evidence',
    'inputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (events_path, rows_path)},
    'rows': len(rows), 'phonetic_blocks_accounted_once': len(used_p),
    'target_answers_accounted_once': len(used_answers),
    'bare_gloss_correction_verified': True,
    'blank_free_translations': [r['source_unit_key'] for r in rows if not r['gloss']],
    'limitation': 'Checks complete native transfer and grouping, not independent visual glyph acceptance or word alignment',
    'canonical_changes': False,
}
(HERE / 'whole-transfer-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
