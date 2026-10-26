"""Join all reverse-review findings to frozen cells, without canonical approval."""
import copy
import json
from collections import Counter
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent
FIELDS = {'chakma_response': 'raw_source_form', 'bengali_prompt': 'raw_bengali_prompt'}


def main():
    source = ROOT / 'reverse-first-reading.jsonl'
    freeze_path = ROOT / 'reverse-first-reading-freeze-20260927.json'
    report_path = ROOT / 'independent-reverse-complete-review-20260927.json'
    caution_path = ROOT / 'independent-ha-glyph-caution-20260927.json'
    freeze, report = json.loads(freeze_path.read_text()), json.loads(report_path.read_text())
    assert sha(freeze_path) == report['freeze_sha256']
    assert sha(source) == freeze['first_reading_sha256']
    for relative, expected in freeze['input_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    assert not report['pending_pages']
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    coverage = {r['source_unit_key']: r for r in report['coverage']}
    assert len(rows) == len(coverage) == report['reviewed_cells'] == report['total_cells'] == 741
    assert set(coverage) == {r['source_unit_key'] for r in rows}
    result = copy.deepcopy(rows)
    findings, held_h, changed_fields = [], [], []
    for row in result:
        review = coverage[row['source_unit_key']]
        assert all(row[f] == review[f] for f in ('printed_page', 'pdf_page', 'block', 'row'))
        assert row['raw_source_form'] == review['target_form_reviewed']
        assert row['raw_bengali_prompt'] == review['bengali_prompt_reviewed']
        row['independent_review_evidence'] = review
        row['reviewed_source_form_proposal'] = row['raw_source_form']
        row['reviewed_bengali_prompt_proposal'] = row['raw_bengali_prompt']
        row['font_control_holds'] = []
        seen_fields = set()
        for finding in review['findings']:
            field = FIELDS[finding['field']]
            assert finding['old'] == row[field]
            proposed = finding['proposed']
            finding_key = {'source_unit_key': row['source_unit_key'], **finding}
            findings.append(finding_key)
            if proposed is not None:
                assert field not in seen_fields, (row['source_unit_key'], field)
                seen_fields.add(field)
                dest = 'reviewed_source_form_proposal' if field == 'raw_source_form' else 'reviewed_bengali_prompt_proposal'
                row[dest] = proposed
                if proposed != finding['old']:
                    changed_fields.append((row['source_unit_key'], field))
                # This identifies review candidates, not an automatic decision
                # that any existing source hasanta is false.
                if 'হ্' in proposed and proposed != finding['old']:
                    hold = dict(finding_key, reason='Proposed h-class reading requires ordinary হ font controls; independent caution supersedes confidence of historical report.')
                    row['font_control_holds'].append(hold)
                    held_h.append(hold)
        row['review_status'] = 'complete independent review reconciled; root adjudication and source integration pending'
        assert row['accepted_for_installation'] is False
    assert len(findings) == 95
    assert Counter(f['status'] for f in findings) == {'correction': 37, 'proposed-correction': 26, 'uncertain': 32}
    target = ROOT / 'reverse-reviewed-reading-proposal.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
    status = {
        'status': 'complete 741-cell reconciliation; no canonical approval',
        'physical_cells': len(rows), 'reviewed_fields': 2 * len(rows),
        'per_printed_page': dict(Counter(r['printed_page'] for r in rows)),
        'finding_status_counts': dict(Counter(f['status'] for f in findings)),
        'changed_field_proposals': len(changed_fields),
        'h_class_font_control_holds': held_h,
        'policy': ['Keep Bengali prompts as source glosses; do not invent English.',
                   'Preserve all alternatives and qualifiers.',
                   'Printed57 left3 Hindi comparator remains a separate control witness.',
                   'Repeated forward and reverse attestations retain chapter provenance.',
                   'A reviewer correction classification is not root approval.'],
        'inputs': {p.name: sha(p) for p in (source, freeze_path, report_path, caution_path)},
        'output': {target.name: sha(target)}, 'installed_rows_changed': 0,
        'pending': ['root adjudication of all findings, including h-class font controls',
                    'alternate/comparator integration and source-wide audit',
                    'metadata/profile/focused validation and authorized compiled gates'],
    }
    (ROOT / 'reverse-review-reconciliation-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'physical_cells': len(rows), 'findings': len(findings),
                      'changed_field_proposals': len(changed_fields), 'h_class_holds': len(held_h),
                      'output_sha256': sha(target)}))


if __name__ == '__main__':
    main()
