"""Reconcile source-supported table metadata without canonical installation."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_path = HERE / 'table-reviewed-reading.json'
    review_path = HERE / 'independent-table-metadata-review-20260927.json'
    source = json.loads(source_path.read_text())
    review = json.loads(review_path.read_text())
    registry = ROOT / 'data/tags.py'
    assert sha(source_path) == review['input_sha256']
    assert sha(registry) == review['registered_tags_sha256']
    assert sha(HERE / 'table-prompt-inventory.jsonl') == review['prompt_inventory_sha256']
    tree = ast.parse(registry.read_text())
    allowed = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GRAMMATICAL_TAGS' for t in n.targets))
    proposals = {r['prompt_id']: r for r in review['items']}
    assert len(proposals) == 233
    # These doubts were explicitly resolved by image adjudication. Retain history.
    resolved = {10, 11, 12, 35, 49, 53, 210}
    # Independent review introduced caveats not present in the initial draft field.
    additional_uncertain = {15, 18, 21, 22, 24, 25, 27, 28, 72, 124, 172, 174, 240}
    output = copy.deepcopy(source)
    for row in output['readings']:
        pid = row['prompt_id']
        proposal = proposals[pid]
        assert row['forms'] == proposal['source_forms'], pid
        assert row.get('tags', []) == proposal['old_tags'], pid
        tags = list(proposal['proposed_tags'])
        if pid in resolved:
            tags = [t for t in tags if t != 'uncertain']
        if pid in additional_uncertain and 'uncertain' not in tags:
            tags.append('uncertain')
        assert len(set(tags)) == len(tags), pid
        assert not set(tags) - allowed, (pid, set(tags) - allowed)
        row['tags'] = tags
        row['metadata_review'] = proposal
        row['source_notes_additions'] = proposal['notes_additions']
        row['current_uncertainty_tag'] = 'uncertain' in tags
        if pid in resolved:
            row['uncertainty_disposition'] = 'Initial question resolved by independent image adjudication; original field preserved as history.'
        elif pid in additional_uncertain:
            row['uncertainty_disposition'] = 'Independent review caveat remains current; see pinned independent review evidence and owner adjudication.'
        row['canonical_emission_authorized'] = False
    keyed = {r['prompt_id']: r for r in output['readings']}
    assert 'sentential' not in keyed[241]['tags']
    assert all('sentential' in keyed[p]['tags'] for p in range(220,241))
    assert all('imp' not in r['tags'] and 'prs' not in r['tags'] for r in keyed.values())
    assert all('du' not in keyed[p]['tags'] for p in [105,114,123])
    output['status'] = 'reviewed literal and metadata candidates; whole-source profile/audit/installation pending'
    output['metadata_reconciliation'] = {
        'review_sha256': sha(review_path), 'input_sha256': sha(source_path),
        'registry_sha256': sha(registry), 'resolved_initial_uncertainties': sorted(resolved),
        'additional_current_uncertainty_tags': sorted(additional_uncertain),
        'policy': 'Keep prompt-scoped grammar; no inferred grammatical gender from animal sex, no inferred nonfinite subtype, no dual from two, and no sentence tag on answer fragment241.',
    }
    path = HERE / 'table-metadata-candidate.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    report = {'status':output['status'], 'written_cells':len(keyed),
              'unknown_tags':[], 'current_uncertain_cells':sum(r['current_uncertainty_tag'] for r in keyed.values()),
              'literal_changes':0, 'canonical_changes':False, 'output_sha256':sha(path),
              **output['metadata_reconciliation']}
    (HERE / 'table-metadata-status.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
