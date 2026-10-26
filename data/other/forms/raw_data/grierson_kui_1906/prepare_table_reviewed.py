"""Preserve the first reading while applying independently adjudicated corrections.

Produces source-stage candidates only. No canonical exporter or database is run.
"""
import copy
import hashlib
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw_path = HERE / 'table-first-reading.json'
    raw = json.loads(raw_path.read_text())
    review_path = HERE / 'independent-table-26-241-review-20260927.json'
    review = json.loads(review_path.read_text())
    assert digest(raw_path) == review['input']['sha256']
    assert digest(ROOT / review['prompt_input']['path']) == review['prompt_input']['sha256']
    for witness in review['sources'] + review['detail_images']:
        assert digest(ROOT / witness['path']) == witness['sha256'], witness['path']
    decisions_path = HERE / 'root-table-correction-decisions-20260927.json'
    decisions = json.loads(decisions_path.read_text())
    accepted = {r['prompt_id']: r for r in decisions['decisions']}
    corrections = {r['prompt_id']: r for r in review['corrections']}
    assert set(accepted) == set(corrections) and len(accepted) == 13
    assert not decisions['remaining_proposed_corrections']
    reviewed = {r['prompt_id']: r for r in review['items']}
    assert set(reviewed) == set(range(26, 242))
    first25 = json.loads((HERE / 'independent-table-first25-review-20260927.json').read_text())
    first25 = {r['prompt_id']: r for r in first25['items']}
    recovery_path = HERE / 'table157-recovery-review-20260927.json'
    recovery = json.loads(recovery_path.read_text())
    assert digest(ROOT / recovery['image']) == recovery['image_sha256']
    assert recovery['accepted_reading'] == 'Inu māñji'
    output = copy.deepcopy(raw)
    for row in output['readings']:
        pid = row['prompt_id']
        original = copy.deepcopy(row)
        evidence = first25[pid] if pid <= 25 else reviewed[pid]
        assert row['literal'] == evidence['old_literal'], pid
        if pid > 25:
            assert row['forms'] == evidence['old_forms'], pid
        row['first_reading'] = original
        row['independent_review'] = evidence
        row['canonical_emission_authorized'] = False
        if pid in accepted:
            proposal = corrections[pid]
            assert accepted[pid]['reading'] == proposal['proposed_literal'], pid
            row['literal'] = proposal['proposed_literal']
            row['forms'] = proposal['proposed_forms']
            row['owner_adjudication'] = accepted[pid]
            # Original uncertainty remains available, even when a component is resolved.
            row['uncertainty_history'] = original.get('uncertainty')
        if pid == 157:
            row['literal'] = recovery['accepted_reading']
            row['forms'] = [recovery['accepted_reading']]
            row['uncertainty_history'] = original.get('uncertainty')
            row['uncertainty'] = recovery['retained_uncertainty']
            row['recovery_review'] = recovery
        for form in row['forms']:
            assert form and '\ufffd' not in form
            assert unicodedata.normalize('NFC', form) == form, (pid, form)
    assert len(output['readings']) == 233
    output['status'] = 'all written cells independently read; corrected candidate transcription; metadata and whole-source gates pending'
    output['input_hashes'] = {
        raw_path.name: digest(raw_path), review_path.name: digest(review_path),
        decisions_path.name: digest(decisions_path), recovery_path.name: digest(recovery_path),
    }
    output['metadata_proposals_pending'] = review['metadata_proposals']
    path = HERE / 'table-reviewed-reading.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    report = {
        'status': output['status'], 'written_cells': 233,
        'independent_reviewed_written_cells': len(first25) + review['coverage']['literal_cells'],
        'adjudicated_corrections': len(accepted),
        'unresolved_headless_candidates': [r['prompt_id'] for r in output['readings'] if not r['forms']],
        'source_blank_prompts': [133, 136], 'future_reference_prompts': list(range(195, 201)),
        'canonical_changed': False, 'output_sha256': digest(path),
        'recovered_from_second_scan': [157],
        'pending': ['metadata/tag reconciliation', 'full symbol/profile review', 'whole-source audit and installation gates'],
    }
    (HERE / 'table-reviewed-status.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
