"""Reconcile both independent passes without losing superseded glyph evidence."""
import copy
import csv
import json
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent


def reconcile(name, expected):
    source = ROOT / f'{name}-first-reading.jsonl'
    manual = ROOT / f'{name}-manual-first-reading.tsv'
    base = ROOT / f'independent-{name}-complete-review-20260927.json'
    extra = ROOT / f'independent-{name}-hasanta-class-addendum-20260927.json'
    review, addendum = json.loads(base.read_text()), json.loads(extra.read_text())
    assert sha(source) == review['inputs'][source.name]
    assert sha(manual) == review['inputs'][manual.name] == addendum['owner_input_sha256']
    assert sha(base) == addendum['base_review_sha256']
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    with manual.open() as stream:
        manual_rows = {(int(r['pdf_page']), int(r['row'])): r for r in csv.DictReader(stream, delimiter='\t')}
    coverage = {r['source_unit_key']: r for r in review['coverage']}
    assert len(rows) == len(coverage) == len(manual_rows) == expected == review['coverage_count']
    assert set(coverage) == {r['source_unit_key'] for r in rows}
    records = copy.deepcopy(rows)
    witnesses = {}
    for row in records:
        r = coverage[row['source_unit_key']]
        pair = (row['pdf_page'], row['physical_row'])
        assert pair == (r['pdf_page'], r['physical_row'])
        assert r['owner_first_reading'] == manual_rows[pair]
        row['independent_review_evidence'] = r
        row['review_status'] = 'both independent passes reconciled; remaining glyph holds and whole-book integration pending'
        for w in row['witnesses']:
            field = w['witness_key'].rsplit(':', 1)[1]
            key = (*pair, field)
            assert key not in witnesses
            assert w['raw_owner_first_reading'] == manual_rows[pair][field]
            w['reviewed_reading_proposal'] = w['raw_owner_first_reading']
            w['review_changes'] = []
            w['review_holds'] = []
            witnesses[key] = w
    assert len(witnesses) == 2 * expected == review['source_witnesses']

    findings = list(review.get('findings', []))
    if name == 'santali':
        for r in review['coverage']:
            findings.extend(dict(f, pdf_page=r['pdf_page'], row=r['physical_row']) for f in r['findings'])
    for origin, proposals in ((base.name, findings), (extra.name, addendum['proposals'])):
        for p in proposals:
            key = (p['pdf_page'], p['row'], p['field'])
            w = witnesses[key]
            if 'proposed' not in p:
                w['review_holds'].append(dict(p, report=origin))
                continue
            assert p['old'] == w['raw_owner_first_reading']
            prior = w['reviewed_reading_proposal']
            w['reviewed_reading_proposal'] = p['proposed']
            w['review_changes'].append(dict(p, report=origin,
                previous_proposal=prior, supersedes_different_proposal=prior not in (p['old'], p['proposed'])))
    # Keep every residual statement even if the addendum gives no field-specific
    # key. A later proposal never silently clears an earlier uncertainty.
    residual = {k: v for k, v in addendum.items()
                if k not in ('proposals', 'base_review_sha256', 'owner_input_sha256')}
    changed = [w for w in witnesses.values() if w['reviewed_reading_proposal'] != w['raw_owner_first_reading']]
    target = ROOT / f'{name}-reviewed-reading-proposal.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    status = {
        'status': 'complete independent-review reconciliation; not canonical approval',
        'physical_rows': expected, 'main_witnesses': len(witnesses),
        'changed_witness_proposals': len(changed),
        'proposal_events_including_repeated_evidence': sum(len(w['review_changes']) for w in witnesses.values()),
        'superseded_proposals': [w['witness_key'] for w in witnesses.values()
                                if any(c['supersedes_different_proposal'] for c in w['review_changes'])],
        'complete_addendum_residual_evidence': residual,
        'base_scope_and_notation_evidence': {k: review[k] for k in
            ('meaning_scope_checks', 'meaning_scopes', 'notation_checks', 'identity_review', 'systematic_class') if k in review},
        'inputs': {p.name: sha(p) for p in (source, manual, base, extra)},
        'output': {target.name: sha(target)}, 'installed_rows_changed': 0,
        'pending': ['root glyph adjudication with all unresolved readings retained',
                    'whole-book integration and metadata/profile checks', 'authorized compiled gates'],
    }
    (ROOT / f'{name}-review-reconciliation-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: status[k] for k in ('physical_rows', 'main_witnesses', 'changed_witness_proposals',
                     'proposal_events_including_repeated_evidence', 'superseded_proposals', 'installed_rows_changed')}))


if __name__ == '__main__':
    reconcile('brokskat', 72)
    reconcile('santali', 96)
