"""Reconcile independent grammar findings as proposals, never installation approval."""
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = 'independent-grammar-full-review-20260927.json'
SOURCE = 'grammar-review-candidate.jsonl'
EXPECTED = {'candidate_tags': 303, 'source_note': 41, 'gloss_bn': 3, 'form': 1}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def reconcile(root=ROOT):
    report = json.loads((root / REPORT).read_text())
    for name, expected in report['input_sha256'].items():
        assert sha(root / name) == expected, f'Frozen input changed: {name}'
    rows = [json.loads(line) for line in (root / SOURCE).read_text().splitlines()]
    by = {r['source_unit_key']: r for r in rows}
    coverage = {r['source_unit_key']: r for r in report['checks']}
    assert len(rows) == len(by) == len(report['checks']) == len(coverage) == 682
    assert set(by) == set(coverage)
    assert len({r['block_id'] for r in rows}) == report['reviewed_blocks'] == 57
    assert len(report['paradigms_reviewed']) == 8
    assert report['reviewed_pages'] == list(range(7, 21))
    assert Counter(c['field'] for c in report['corrections']) == EXPECTED
    changes, uncertainties = defaultdict(list), defaultdict(list)
    seen = set()
    for c in report['corrections']:
        key, field = c['source_unit_key'], c['field']
        assert (key, field) not in seen, 'Duplicate field proposal'
        seen.add((key, field))
        assert by[key][field] == c['old'], (key, field, 'Old value mismatch')
        assert c['old'] != c['new']
        if field == 'candidate_tags':
            assert set(c['old']) <= set(c['new']), 'Review must not discard existing tags'
            assert len(c['new']) == len(set(c['new']))
        changes[key].append(c)
    for u in report['uncertainties']:
        assert u['source_unit_key'] in by
        assert u['accepted_correction'] is False
        uncertainties[u['source_unit_key']].append(u)
    assert len(report['uncertainties']) == 30
    result = []
    for original in rows:
        key = original['source_unit_key']; check = coverage[key]
        for field, review_field in [('form','form_reviewed'), ('gloss_bn','gloss_bn_reviewed'), ('candidate_tags','candidate_tags_reviewed')]:
            assert original[field] == check[review_field], (key, field)
        assert original['pdf_page'] == check['pdf_page']
        assert original['block_id'] == check['block_id']
        row = copy.deepcopy(original)
        row['reviewed_fields_proposal'] = {field: copy.deepcopy(original[field]) for field in EXPECTED}
        for c in changes[key]:
            row['reviewed_fields_proposal'][c['field']] = copy.deepcopy(c['new'])
        row['independent_review_evidence'] = copy.deepcopy(check)
        row['independent_review_changes'] = copy.deepcopy(changes[key])
        row['independent_typed_uncertainties'] = copy.deepcopy(uncertainties[key])
        row['owner_review_status'] = original.get('review_status', '')
        row['accepted_for_installation'] = False
        row['review_status'] = 'independent proposals reconciled; root adjudication and whole-source integration pending'
        # Preserve all original fields and owner uncertainties, even when a proposal differs.
        assert all(row[k] == v for k, v in original.items() if k != 'review_status')
        result.append(row)
    status = {
        'status': 'complete grammar review reconciliation; proposals are not accepted readings',
        'records': 682, 'blocks': 57, 'paradigms': report['paradigms_reviewed'],
        'field_recommendations': EXPECTED, 'changed_keys': sum(bool(c) for c in changes.values()),
        'independent_typed_uncertainties': 30,
        'owner_uncertainty_records': sum(bool(r.get('uncertainties')) for r in rows),
        'scope_notes': report['scope_notes'],
        'inputs': {name: sha(root / name) for name in [*report['input_sha256'], REPORT]},
        'reconciler_sha256': sha(Path(__file__)),
        'installed_rows_changed': 0,
        'pending': ['Root adjudication of each proposed correction and typed uncertainty',
                    'Ordinary হ font controls before accepting h-hasanta readings',
                    'Whole-source integration, fresh output audit, metadata/profile and authorized validation gates'],
    }
    return result, status

def main():
    rows, status = reconcile()
    target = ROOT / 'grammar-reviewed-reading-proposal.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    status['output'] = {target.name: sha(target)}
    (ROOT / 'grammar-review-reconciliation-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: status[k] for k in ['records','blocks','field_recommendations','changed_keys','independent_typed_uncertainties','installed_rows_changed','output']}))

if __name__ == '__main__':
    main()
