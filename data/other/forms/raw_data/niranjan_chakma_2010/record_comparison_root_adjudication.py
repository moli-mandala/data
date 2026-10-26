"""Root adjudication of Kokborok and further-table review proposals."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = {
    'kokborok': '1bfff190652a69cce81d3e13157f9ddebb6a4aeb94db4095c06e5abbb62857fb',
    'further': 'f33fedf61de3bccc74848a8eb1bcd4ab697f195c5d9f8e6d9e6168ed3c1d6b91',
}
HELD = {'ভাহ্‌ঝ', 'মালাহ্', 'গাহ্‌ক', 'থেধাহ্', 'গোহ্‌র', 'কুন্‌থি'}
decisions, inputs = [], {}
for section, digest in REPORTS.items():
    report_path = ROOT / f'independent-{section}-full-review-20260927.json'
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == digest
    report = json.loads(report_path.read_text())
    inputs[report_path.name] = digest
    source = ROOT / f'{section}-first-reading.jsonl'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report['input_sha256'][source.name]
    inputs[source.name] = report['input_sha256'][source.name]
    originals = {w['witness_key']: w for line in source.read_text().splitlines()
                 for w in json.loads(line)['witnesses']}
    for category in ('corrections', 'rendering_only_recommendations', 'label_rendering_recommendations'):
        for change in report.get(category, []):
            field = 'source_language_label' if category == 'label_rendering_recommendations' else 'raw_owner_first_reading'
            assert originals[change['witness_key']][field] == change['old']
            hold = change['new'] in HELD
            decisions.append({
                'source_unit_key': change['source_unit_key'], 'witness_key': change['witness_key'],
                'field': field, 'old': change['old'], 'proposed_value': change['new'],
                'accepted_value': None if hold else change['new'],
                'root_decision': 'typed glyph hold' if hold else 'accepted literal/rendering correction',
                'evidence': change.get('evidence', change.get('reason')),
                'root_review': 'Root inspected original PDF90–92 and PDF100,101,102 page renders and compared the keyed proposals. Unicode joining controls preserve visibly separated letters; no pronunciation inferred.',
                'hold_reason': ('Dental/retroflex consonant identity still uncertain.' if change['new'] == 'কুন্‌থি'
                                else 'Real additional mark on হ; intended encoding remains uncertain.') if hold else None,
            })
assert len(decisions) == 43
assert sum(d['accepted_value'] is not None for d in decisions) == 36
result = {'input_sha256': inputs, 'decisions': decisions, 'accepted_fields': 36,
          'held_fields': 7, 'canonical_changes': False, 'whole_source_acceptance': False,
          'policy': 'Controls remain controls; label rendering does not prove a language identity. Preserve all independent-review uncertainties, source-specific glosses, alternate readings and original fields.'}
(ROOT / 'comparison-root-adjudication-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'accepted_fields': 36, 'held_fields': 7, 'canonical_changes': False}))
