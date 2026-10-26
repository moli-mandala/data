"""Approve supported target identities while keeping physical and family scope distinct."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / 'numbered-final-identity-review-proposal.json'
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == 'e94003b9dcef33cf9e92120475d8eb631bfc28e360993a55c1a2c3fd629d95ce'
proposal = json.loads(path.read_text())
registry_path = ROOT.parents[4] / 'cldf/languages.csv'
with registry_path.open() as handle:
    registry = {r['ID']: r for r in csv.DictReader(handle)}
decisions = []
for row in proposal['comparators']:
    base = row['proposed_base_id']
    physical = row['original_disposition']
    scope = row['scope_disposition']
    accepted = None
    if physical == 'explicit source blank':
        disposition = 'explicit blank; audit only'
    elif physical == 'unresolved label/form segmentation':
        disposition = 'segmentation hold; audit only'
    elif scope == 'outside requested families; retain comparison control':
        disposition = 'outside-family comparison control; audit only'
    elif scope == 'target-family identity proposal' and base == 'Chenchu':
        disposition = 'identified Chenchu; registry addition pending'
    elif scope == 'target-family identity proposal' and base:
        assert base in registry
        for field in ('ID', 'Name', 'Glottocode', 'Clade'):
            assert registry[base][field] == row['registry_record'][field]
        if base == 'Pk' or row['identity_status'] == 'base supported subject to preserving Vedic source stage':
            assert row['source_stage_or_variety'] and row['required_source_note']
        accepted = base
        disposition = 'accepted existing target-family base'
    else:
        disposition = 'unresolved attribution; audit only'
    decisions.append({'source_unit_key': row['comparison_key'], 'raw_label': row['raw_label'],
                      'literal_form': row['literal_form'], 'accepted_language_id': accepted,
                      'proposed_language_id': base, 'disposition': disposition,
                      'source_stage_or_variety': row['source_stage_or_variety'],
                      'required_source_note': row['required_source_note'],
                      'work_provenance': row['work_provenance'],
                      'evidence': row['evidence'], 'decision_basis': row['decision_basis'],
                      'identity_uncertainty': row['identity_status'] if accepted is None else None,
                      'graph_relation': 'comparison only; no ancestry or borrowing inferred'})
assert len(decisions) == len({d['source_unit_key'] for d in decisions}) == 395
counts = Counter(d['disposition'] for d in decisions)
assert counts['accepted existing target-family base'] == 209
assert counts['outside-family comparison control; audit only'] == 37
assert counts['explicit blank; audit only'] == 2
assert counts['segmentation hold; audit only'] == 1
assert counts['identified Chenchu; registry addition pending'] == 4
result = {'input_sha256': {path.name: digest},
          'registry_sha256': hashlib.sha256(registry_path.read_bytes()).hexdigest(),
          'decisions': decisions, 'counts': dict(counts), 'canonical_changes': False,
          'policy': 'Complete physical coverage does not make every comparator an in-scope lexical attestation. Preserve all controls and unresolved labels in the audit. An approved base does not certify a glyph, identify a narrower dialect, or create a graph edge.'}
(ROOT / 'numbered-identity-root-adjudication-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(counts)))
