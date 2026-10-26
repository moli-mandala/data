"""Accept supported reverse proposals, explicitly withdrawing false insertions."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / 'reverse-field-recheck/complete63-field-proposal-recheck-20260927.json'
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == 'a3d45248f74b2984c859d8a0b54cd9f4ba739c532073bc54059986e11a360c19'
report = json.loads(path.read_text())
source = ROOT / 'reverse-reviewed-reading-proposal.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest() == report['reconciled_input_sha256']
rows = {r['source_unit_key']: r for r in map(json.loads, source.read_text().splitlines())}
assert len(rows) == 741
decisions = []
for check in report['coverage']:
    key, field, kind = check['source_unit_key'], check['field'], check['decision']
    source_field = {'chakma_response': 'raw_source_form', 'bengali_prompt': 'raw_bengali_prompt'}[field]
    assert rows[key][source_field] == check['old']
    if kind == 'reject-proposal':
        accepted = check['old']
        decision = 'rejected insertion; retain original reading'
    elif kind == 'new-reading-hold':
        accepted = None
        decision = 'retain original provisionally with typed field uncertainty'
    else:
        assert kind.startswith('confirmed')
        accepted = check['earlier_proposed']
        decision = 'accepted reviewed correction'
    uncertain = kind in ('new-reading-hold', 'confirmed-with-typed-ha-uncertainty', 'confirmed-with-existing-uncertainty')
    decisions.append({'source_unit_key': key, 'field': source_field, 'review_field': field, 'old': check['old'],
                      'accepted_value': accepted, 'proposed_value': check['earlier_proposed'],
                      'root_decision': decision, 'pdf_page': check['pdf_page'],
                      'required_tags': ['uncertain'] if uncertain else [],
                      'typed_uncertainty': ('marked-ha-encoding' if kind == 'confirmed-with-typed-ha-uncertainty'
                                            else 'gloss-reading' if field == 'bengali_prompt' and uncertain
                                            else 'transcription' if uncertain else None),
                      'evidence': check['note'], 'image_evidence': 'reverse-field-recheck/' + check['evidence'],
                      'root_basis': 'Root reviewed every keyed recheck outcome against the earlier proposal and source-wide font policy; original-image reread performed by the independent reviewer of the first transcription, with same-author recheck disclosed.'})
assert len(decisions) == len({(d['source_unit_key'], d['field']) for d in decisions}) == 63
counts = Counter(r['decision'] for r in report['coverage'])
assert counts == {'confirmed': 43, 'confirmed-with-typed-ha-uncertainty': 10,
                  'confirmed-with-existing-uncertainty': 2, 'reject-proposal': 3, 'new-reading-hold': 5}
rejected = {d['accepted_value'] for d in decisions if d['root_decision'].startswith('rejected')}
assert rejected == {'খাজনা/খাজানা', 'হোচবেঙ', 'কোচপানা'}
result = {'input_sha256': {str(path.relative_to(ROOT)): digest, source.name: report['reconciled_input_sha256']},
          'decisions': decisions, 'accepted_changes': 55, 'rejected_changes': 3, 'held_changes': 5,
          'mandatory_uncertainty_fields': 17, 'canonical_changes': False,
          'policy': 'All earlier residual uncertainties remain. Five held proposals do not authorize dropping their records or silently installing the proposed strings. Preserve first reading, competing candidates and image evidence; prompt-only uncertainty is distinct from target transcription uncertainty.'}
(ROOT / 'reverse-root-adjudication-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('accepted_changes', 'rejected_changes', 'held_changes')}))
