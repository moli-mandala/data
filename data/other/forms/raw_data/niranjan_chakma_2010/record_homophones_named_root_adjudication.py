"""Record root's original-image adjudication, leaving broader review gates explicit."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    'homophones': {(73, 1): 'পেল', (73, 2): 'পেল', (87, 2): 'চিত্',
                   (99, 1): 'ঝাঁতি,', (127, 2): 'জেঠ', (128, 2): 'দুত্',
                   (145, 2): 'কুঝি', (149, 3): 'ভভা'},
    'named-groups': {(1, 1): 'আক্‌চোনা', (1, 2): 'অক্‌চানা', (13, 1): 'চরাং',
                     (25, 1): 'নাগ্‌চা', (26, 2): 'লুদি নাক্‌স ফুল',
                     (28, 1): 'থুনি', (33, 1): 'তেই', (60, 2): 'ফেনক বাত্তি',
                     (64, 1): 'নানু, বেই, বেভেই\nবেই, নানু, মেঁমেই', (70, 1): 'পেন চলং'},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    decisions, sample, inputs = [], [], {}
    for section, expected in EXPECTED.items():
        path = ROOT / f'independent-{section}-complete-review-20260927.json'
        report = json.loads(path.read_text())
        inputs[path.name] = sha(path)
        rows = {r['source_unit_key']: r for r in map(json.loads, (ROOT / f'{section}-first-reading.jsonl').read_text().splitlines())}
        for filename, digest in report['input_sha256'].items():
            assert sha(ROOT / filename) == digest
        seen = set()
        for row in report['coverage']:
            for finding in row['findings']:
                if finding['status'] != 'correction':
                    continue
                identity = row['printed_group'], row['group_cell']
                assert identity not in seen
                seen.add(identity)
                assert finding['proposed'] == expected[identity]
                assert rows[row['source_unit_key']][finding['field']] == finding['old']
                decisions.append({'source_unit_key': row['source_unit_key'],
                                  'pdf_page': row['pdf_page'], 'printed_page': row['printed_page'],
                                  **finding, 'root_decision': 'accepted literal correction',
                                  'evidence': 'Root independently inspected original PDF page renders at 3x; preserves source lettering rather than modernizing spelling.'})
        assert seen == set(expected)
        eligible = sorted((r for r in report['coverage'] if not r['findings']), key=lambda r: r['source_unit_key'])
        for row in random.Random('20260927-root-' + section).sample(eligible, 10):
            sample.append({'section': section, **row, 'root_audit_status': 'pending original-image comparison'})
    result = {'reviewer': 'root', 'independent_of_first_reading': True,
              'original_pdf_sha256': report['original_pdf_sha256'], 'input_sha256': inputs,
              'accepted_literal_corrections': len(decisions), 'decisions': decisions,
              'canonical_changes': False, 'accepted_for_installation': False,
              'pending': ['fresh seeded audit', 'remaining proposed and uncertain readings',
                          'whole-source integration and authorized validation gates']}
    (ROOT / 'homophones-named-root-adjudication-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    sample_path = ROOT / 'homophones-named-root-fresh-sample-20260927.json'
    # Never overwrite completed manual audit decisions on a reproducibility rerun.
    if not sample_path.exists():
        sample_path.write_text(json.dumps({'seed': '20260927-root-<section>', 'sample': sample}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'accepted_corrections': len(decisions), 'fresh_sample': len(sample), 'canonical_changes': False}))


if __name__ == '__main__':
    main()
