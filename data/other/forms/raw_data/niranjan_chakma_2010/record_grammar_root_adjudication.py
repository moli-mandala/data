"""Adjudicate the independently reviewed grammatical metadata without settling glyph holds."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
report_path = ROOT / 'independent-grammar-full-review-20260927.json'
candidate_path = ROOT / 'grammar-review-candidate.jsonl'
PINNED = {
    report_path.name: '4ca2fdc34890b0cdc98ded6490245b734d35d6d7cf756dc514b4c4f582ccc1fd',
    candidate_path.name: '42b1e04d15c42b6f70e4c1fc1e34d594cba5e43ab7ee49a18046bf1ab2dea982',
}
for name, digest in PINNED.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
report = json.loads(report_path.read_text())
rows = {r['source_unit_key']: r for r in map(json.loads, candidate_path.read_text().splitlines())}
assert len(rows) == 682
expected = {'candidate_tags': 303, 'source_note': 41, 'gloss_bn': 3, 'form': 1}
assert Counter(r['field'] for r in report['corrections']) == expected
decisions = []
seen = set()
conflicts = {
    'niranjan2010:p13:grammar:b4:subject:3',
    'niranjan2010:p14:grammar:b3:subject:1',
    'niranjan2010:p14:grammar:b3:subject:4',
}
for correction in report['corrections']:
    key, field = correction['source_unit_key'], correction['field']
    assert (key, field) not in seen
    seen.add((key, field))
    assert rows[key].get(field, '') == correction['old']
    held = field == 'form'
    if field == 'candidate_tags':
        assert key not in conflicts, 'Do not assign an unqualified contradictory person tag.'
        assert set(correction['old']) <= set(correction['new'])
    decisions.append({
        'source_unit_key': key, 'field': field, 'old': correction['old'],
        'accepted_value': None if held else correction['new'],
        'proposed_value': correction['new'],
        'root_decision': 'held for glyph adjudication' if held else 'accepted metadata correction',
        'evidence': correction['evidence'], 'pdf_page': correction['pdf_page'],
        'basis': 'Independent exhaustive original-page review; root inspected correction categories, all tag transformation classes, all note changes, gloss splits and guards for contradictory pronoun cells. This is not a claim of another complete root image rereading.',
    })
result = {
    'input_sha256': PINNED, 'decisions': decisions,
    'accepted_metadata_fields': 347, 'held_literal_fields': 1,
    'policy': 'Preserve all original forms and source provenance, all 30 independent typed uncertainties and all owner uncertainties. Source headings scope tags only where explicit; contradictory pronoun cells receive notes rather than unqualified new person tags. No reconstructed paradigm cells or graph relations.',
    'canonical_changes': False, 'whole_source_acceptance': False,
}
(ROOT / 'grammar-root-adjudication-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'accepted_metadata_fields': 347, 'held_literal_fields': 1, 'records': len(rows)}))
