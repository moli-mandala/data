"""Carry keyed review proposals forward, preserving frozen originals and all holds."""
import copy
import json
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent
TABLES = [('kokborok', 68, 'independent-kokborok-full-review-20260927.json'),
          ('further', 84, 'independent-further-full-review-20260927.json')]


def reconcile(name, expected, report_name):
    source = ROOT / f'{name}-first-reading.jsonl'
    report_path = ROOT / report_name
    report = json.loads(report_path.read_text())
    assert sha(source) == report['input_sha256'][source.name]
    originals = list(map(json.loads, source.read_text().splitlines()))
    coverage = {r['source_unit_key']: r for r in report['coverage']}
    assert len(originals) == len(coverage) == expected == report['rows_reviewed']
    assert set(coverage) == {r['source_unit_key'] for r in originals}
    records = copy.deepcopy(originals)
    witnesses = {}
    for row in records:
        review = coverage[row['source_unit_key']]
        assert (row['pdf_page'], row['physical_row']) == (review['pdf_page'], review['physical_row'])
        rw = {w['witness_key']: w for w in review['witnesses']}
        assert set(rw) == {w['witness_key'] for w in row['witnesses']}
        row['independent_review_evidence'] = review
        row['review_status'] = 'complete independent review reconciled; root adjudication and whole-book integration pending'
        for witness in row['witnesses']:
            key = witness['witness_key']
            assert key not in witnesses
            assert witness['raw_owner_first_reading'] == rw[key]['old']
            witness['reviewed_reading_proposal'] = witness['raw_owner_first_reading']
            witness['reviewed_label_proposal'] = witness['source_language_label']
            witness['review_changes'] = []
            witnesses[key] = witness
    seen = set()
    for correction in report['corrections']:
        key = correction['witness_key']
        assert key not in seen
        seen.add(key)
        witness = witnesses[key]
        assert witness['raw_owner_first_reading'] == correction['old']
        assert correction['field'] == 'raw_owner_first_reading'
        witness['reviewed_reading_proposal'] = correction['new']
        witness['review_changes'].append(dict(correction, proposal_type='literal correction'))
    for correction in report.get('rendering_only_recommendations', []):
        witness = witnesses[correction['witness_key']]
        assert witness['reviewed_reading_proposal'] == correction['old']
        witness['reviewed_reading_proposal'] = correction['new']
        witness['review_changes'].append(dict(correction, proposal_type='rendering only'))
    for correction in report.get('label_rendering_recommendations', []):
        witness = witnesses[correction['witness_key']]
        assert witness['source_language_label'] == correction['old']
        witness['reviewed_label_proposal'] = correction['new']
        witness['review_changes'].append(dict(correction, proposal_type='label rendering only; identity unchanged'))
    assert len(seen) == report['material_field_corrections']
    # Every coverage proposal must agree with its complete correction list.
    for review in coverage.values():
        for witness in review['witnesses']:
            key = witness['witness_key']
            original = witnesses[key]['raw_owner_first_reading']
            proposed = witness['proposed']
            assert proposed == original or any(c['new'] == proposed for c in witnesses[key]['review_changes'])
    target = ROOT / f'{name}-reviewed-reading-proposal.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    status = {
        'status': 'review proposals reconciled; not canonical approval', 'physical_rows': expected,
        'main_witnesses': len(witnesses), 'literal_field_corrections': len(seen),
        'rendering_only_changes': len(report.get('rendering_only_recommendations', [])),
        'label_rendering_only_changes': len(report.get('label_rendering_recommendations', [])),
        'unresolved_and_scope_findings': {k: report[k] for k in ('unresolved','scope_review','scope_findings','source_language_identity_review') if k in report},
        'inputs': {source.name:sha(source), report_path.name:sha(report_path)},
        'output': {target.name:sha(target)}, 'installed_rows_changed': 0,
        'pending': ['root original-glyph adjudication of corrections and residual uncertainties',
                    'identity/reference/alternate/grammar integration', 'fresh independent whole-source audit and validation gates'],
    }
    (ROOT/f'{name}-review-reconciliation-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:status[k] for k in ('physical_rows','main_witnesses','literal_field_corrections','rendering_only_changes','label_rendering_only_changes','installed_rows_changed')}, ensure_ascii=False))


if __name__ == '__main__':
    for args in TABLES:
        reconcile(*args)
