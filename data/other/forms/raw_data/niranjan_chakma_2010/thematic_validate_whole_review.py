"""Validate coverage and candidate applicability without changing canonical data."""
import csv
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
CANONICAL = PACKAGE.parent.parent / '20260921-niranjan-chakma.csv'

def validate():
    report = json.loads((PACKAGE / 'thematic-whole-glyph-review-proposal.json').read_text())
    rows = list(csv.reader(CANONICAL.open()))
    assert len(rows) == 742
    assert hashlib.sha256(CANONICAL.read_bytes()).hexdigest() == report['canonical_sha256']
    keyed = {r[10]: r for r in rows}
    assert len(keyed) == 742
    assert sorted(p['printed_page'] for p in report['pages']) == list(range(19, 47))
    compared = [k for p in report['pages'] for k in p['all_installed_keys_compared']]
    assert len(compared) == len(set(compared)) == 742
    assert set(compared) == set(keyed)
    assert not report['remaining_printed_pages']
    for page in report['pages']:
        assert page['pdf_page'] == page['printed_page'] + 2
        assert page['installed_rows'] == len(page['all_installed_keys_compared'])
        for key, form in page['source_forms_at_review'].items():
            assert keyed[key][2] == form
    changed = report['corrections']
    assert len({c['source_entry_key'] for c in changed}) == len(changed)
    for candidate in changed + report['uncertain_candidates']:
        assert keyed[candidate['source_entry_key']][2] == candidate['before']
    summary = {
        'status': 'All 28 original pages reviewed; candidate corrections require independent adjudication',
        'canonical_rows_unchanged': len(rows),
        'canonical_sha256': report['canonical_sha256'],
        'reviewed_original_pages': len(report['pages']),
        'original_physical_cells': sum(p['original_physical_cells'] for p in report['pages']),
        'installed_physical_cells': sum(p['physical_cells'] for p in report['pages']),
        'withheld_source_cells': [a for p in report['pages'] for a in p['withheld_source_cells']],
        'proposed_corrections': len(changed),
        'unresolved_candidate_readings': len(report['uncertain_candidates']),
        'report_sha256': hashlib.sha256((PACKAGE / 'thematic-whole-glyph-review-proposal.json').read_bytes()).hexdigest(),
        'canonical_edits': False,
    }
    assert summary['original_physical_cells'] == 733
    assert summary['installed_physical_cells'] + len(summary['withheld_source_cells']) == 733
    return summary

if __name__ == '__main__':
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
