"""Preserve complete same-author second readings as keyed review proposals."""
import copy
import json
from collections import Counter
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent


def reconcile(name, count):
    source = ROOT / f'{name}-first-reading.jsonl'
    report_path = ROOT / f'independent-{name}-complete-review-20260927.json'
    report = json.loads(report_path.read_text())
    for filename, expected in report['input_sha256'].items():
        assert sha(ROOT / filename) == expected
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    coverage = {r['source_unit_key']: r for r in report['coverage']}
    assert len(rows) == len(coverage) == report['reviewed_cells'] == report['total_cells'] == count
    assert {r['source_unit_key'] for r in rows} == set(coverage)
    result = copy.deepcopy(rows)
    findings = []
    for row in result:
        review = coverage[row['source_unit_key']]
        assert all(row[k] == review[k] for k in ('printed_page', 'pdf_page', 'printed_group', 'group_cell'))
        assert row['target_form'] == review['form']
        assert row['definition_bn'] == review['definition_bn']
        if name == 'named-groups':
            assert row['source_group_label'] == review['source_group_label']
        row['reviewed_field_proposals'] = {k: row[k] for k in ('target_form', 'definition_bn', 'source_group_label') if k in row}
        row['second_reading_evidence'] = review
        row['reviewer_independent_of_first_reading'] = False
        row['review_method'] = report['review_method']
        changed = set()
        for finding in review['findings']:
            field = finding['field']
            assert field in row['reviewed_field_proposals']
            assert finding['old'] == row[field]
            if finding['proposed'] is not None:
                assert field not in changed
                changed.add(field)
                row['reviewed_field_proposals'][field] = finding['proposed']
            findings.append({'source_unit_key': row['source_unit_key'], **finding})
        row['review_status'] = 'complete same-author second reading reconciled; independent validation and source integration pending'
        row['accepted_for_installation'] = False
    target = ROOT / f'{name}-reviewed-reading-proposal.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
    status = {'status': 'complete review reconciliation; not independent transcription certification',
              'cells': count, 'findings': dict(Counter(f['status'] for f in findings)),
              'reviewer_independent_of_first_reading': False,
              'scope_policy': 'Preserve all homographs, distinct definitions, literal group labels, missing labels, and Tanchangya attributions.',
              'font_policy': 'All h-class readings remain subject to independent-ha-glyph-caution-20260927.json.',
              'inputs': {source.name: sha(source), report_path.name: sha(report_path)},
              'output': {target.name: sha(target)}, 'canonical_changes': False,
              'pending': ['independent correction adjudication and fresh raw-versus-output audit',
                          'typed residual uncertainties and language/dialect mapping',
                          'whole-source integration, focused and authorized compiled gates']}
    (ROOT / f'{name}-review-reconciliation-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'section': name, 'cells': count, 'findings': status['findings'], 'sha256': sha(target)}))


if __name__ == '__main__':
    reconcile('homophones', 427)
    reconcile('named-groups', 145)
