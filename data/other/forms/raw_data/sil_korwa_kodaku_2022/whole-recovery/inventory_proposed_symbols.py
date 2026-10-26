"""Inventory every current candidate literal; this does not approve a profile."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata as ud

HERE = Path(__file__).resolve().parent
FILES = [
    'comparisons/comparison-reviewed-candidate.jsonl',
    'interlinear-review/whole-expression-proposal.jsonl',
    'interlinear-review/c5-column-proposal.jsonl',
    'interlinear-review/continuous-semantic-proposal.jsonl',
    'outside-review/outside-candidate.jsonl',
]
counts = Counter()
clusters = Counter()
examples = defaultdict(list)
issues = []
sources = {}
inputs = {}
for name in FILES:
    path = HERE / name
    raw = path.read_bytes()
    inputs[name] = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    sources[name] = len(rows)
    for row in rows:
        form = row['form']
        key = row.get('entry_key', row.get('source_unit_key'))
        assert key and isinstance(form, str)
        normalized = ud.normalize('NFC', form)
        if normalized != form:
            issues.append({'key': key, 'type': 'non-NFC candidate', 'form': form})
        for c in form:
            if c == '\ufffd' or ud.category(c) in {'Co', 'Cc', 'Cs'}:
                issues.append({'key': key, 'type': 'encoding-or-control',
                               'codepoint': f'U+{ord(c):04X}'})
        counts.update(normalized)
        for c in set(normalized):
            if len(examples[c]) < 4:
                examples[c].append({'key': key, 'form': form})
        pieces = []
        for c in ud.normalize('NFD', form):
            if ud.combining(c) and pieces:
                pieces[-1] += c
            else:
                pieces.append(c)
        clusters.update(pieces)
assert inputs == {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in FILES}, 'Input changed during scan'
inventory = [{'character': c, 'codepoint': f'U+{ord(c):04X}',
              'unicode_name': ud.name(c, 'UNNAMED'), 'count': n,
              'examples': examples[c]} for c, n in sorted(counts.items())]
report = {
    'status': 'Complete snapshot symbol census; includes held and documentary candidates',
    'inputs': inputs, 'candidate_records_by_file': sources,
    'distinct_NFC_characters': len(counts),
    'NFD_combining_clusters': [{'cluster': c, 'count': n} for c, n in sorted(clusters.items())
                              if any(ud.combining(x) for x in c)],
    'encoding_issues': issues,
    'not_proven': ['Lexical acceptance of every candidate', 'Profile mappings',
                  'Final installed normalization', 'Phonemic analysis'],
    'canonical_changes': False,
}
(HERE / 'integration-proposal/symbol-inventory.json').write_text(
    json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
(HERE / 'integration-proposal/symbol-inventory-status.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('inputs', 'NFD_combining_clusters')}, ensure_ascii=False, indent=2))
