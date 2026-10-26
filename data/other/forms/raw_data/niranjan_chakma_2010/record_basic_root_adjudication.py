"""Account for every basic-list cell and distinguish glyph holds from missing records."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / 'basic-reviewed-reading.jsonl'
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == 'b2a2bc3b910febcb8fcf7ff2f3be23e4dd0d1098709fc187b3b287deb7f1e574'
rows = list(map(json.loads, path.read_text().splitlines()))
assert len(rows) == len({r['source_unit_key'] for r in rows}) == 99
AUDIT_ONLY = {
    'niranjan2010chakma:p106:basic:left:row11': 'Simple গ in provisional এগা is not supported by the more complex printed cluster; no certified replacement yet.',
    'niranjan2010chakma:p107:basic:left:row5': 'Damaged medial/final attachment in provisional অজভ’ prevents secure lexical reading; preserve apostrophe and complete source evidence.',
}
decisions = []
for row in rows:
    key = row['source_unit_key']
    held = key in AUDIT_ONLY
    uncertain = bool(row['transcription_hold'])
    decisions.append({'source_unit_key': key, 'field': 'raw_owner_first_reading',
                      'old': row['raw_owner_first_reading'],
                      'accepted_value': None if held else row['reviewed_reading'],
                      'provisional_reading': row['reviewed_reading'],
                      'root_decision': 'audit-only unresolved glyph' if held else ('eligible with typed transcription uncertainty' if uncertain else 'accepted reviewed literal'),
                      'pdf_page': row['pdf_page'], 'printed_english': row['printed_english'],
                      'bengali_control': row['bengali_control'],
                      'required_tags': ['uncertain'] if uncertain and not held else [],
                      'typed_uncertainty': row['typed_remaining_uncertainty'],
                      'evidence': AUDIT_ONLY.get(key, row['independent_review_evidence']),
                      'root_review': 'Root inspected both complete original pages; exhaustive independent keyed review supplies detailed glyph evidence. English/Bengali mismatches remain source claims, not editorially harmonized.'})
assert sum(d['accepted_value'] is not None for d in decisions) == 97
assert sum(bool(d['required_tags']) for d in decisions) == 1
result = {'input_sha256': {path.name: digest}, 'decisions': decisions,
          'source_cells': 99, 'eligible_reviewed': 96, 'eligible_typed_uncertain': 1,
          'audit_only_glyph_holds': 2, 'omitted_unaccounted_cells': 0,
          'canonical_changes': False, 'whole_source_acceptance': False,
          'policy': 'Exact target cells and independent source translations are preserved. Bengali controls are never emitted as Chakma. Dog remains provisional কুত্তর with cluster uncertainty; no substitution from Bengali কুকুর. Two glyph holds remain recoverable in the audit rather than fabricated lexical readings.'}
(ROOT / 'basic-root-adjudication-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('source_cells', 'eligible_reviewed', 'eligible_typed_uncertain', 'audit_only_glyph_holds')}))
