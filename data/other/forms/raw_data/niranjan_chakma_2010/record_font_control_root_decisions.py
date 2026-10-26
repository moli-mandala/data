"""Adopt keyed font evidence while retaining unresolved marked-ha encoding."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / 'h-font-control-review/keyed-font-control-decisions-20260927.json'
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == '9897398788241b7cf052b7ebdbb0ed88b89898f5a9b268457c9c52d1b259f4ec'
report = json.loads(path.read_text())
decisions = []
for row in report['candidate_decisions']:
    if not row.get('conjunct_decision'):
        continue
    assert row['extra_mark_observed'] is True
    assert row['first_reading'].replace('হ্', 'হ্‌') == row['provisional_disjoined_candidate']
    decisions.append({'source_unit_key': row['source_unit_key'], 'field': row['field'],
                      'old': row['first_reading'], 'accepted_value': row['provisional_disjoined_candidate'],
                      'root_decision': 'accepted disjoining with mandatory encoding uncertainty',
                      'evidence': row['conjunct_decision'], 'pdf_page': row['pdf_page'],
                      'required_tags': ['uncertain'], 'typed_uncertainty': 'marked-ha-encoding',
                      'encoding_caveat': row['encoding_caveat']})
assert len(decisions) == 4
result = {'input_sha256': {str(path.relative_to(ROOT)): digest}, 'decisions': decisions,
          'candidate_uncertainty_evidence': report['candidate_decisions'],
          'scope': report['scope'], 'canonical_changes': False,
          'policy': 'Preserve real extra marks and existing provisional encoding with typed uncertainty; no phonological claim. Four disjoining changes do not remove their encoding caveats.',
          'root_review': 'Control atlases 1 and 2 inspected; joining decisions supported by keyed candidate review.'}
(ROOT / 'font-control-root-adjudication-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'accepted_disjoining': 4, 'mandatory_typed_uncertainty': True}))
