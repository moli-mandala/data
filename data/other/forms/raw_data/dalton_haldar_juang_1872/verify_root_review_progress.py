"""Verify frozen identity and record root's original-image review progress."""
import csv
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
O = P / 'whole-source-recovery'
D = P.parents[4]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest_path = O / 'frozen-manifest-20260927.json'
    manifest = json.loads(manifest_path.read_text())
    packet = json.loads((O / 'independent-sample-selection-2026092727.json').read_text())
    assert sha(manifest_path) == packet['frozen_manifest_sha256']
    for name, expected in manifest['files'].items():
        assert sha(P / name) == expected, name
    assert sha(D / manifest['test_file']) == manifest['test_sha256']
    assert sha(Path('/private/tmp/dalton-1872.pdf')) == manifest['source_original_sha256']
    assert sha(D / 'data/other/forms/20260925-dalton-haldar-juang.csv') == manifest['canonical_before_sha256']
    rows = {r[10]: r for r in csv.reader((O / 'proposal.csv').open())}
    for unit in packet['sample_units'] + packet['additional_morphology_edges']:
        assert [rows[k] for k in unit['entry_keys']] == unit['actual_rows']
    for unit in packet['controls']:
        assert not unit['entry_keys'] and unit['source_unit_key'] not in rows
    # Independent readings from full original PDF257 rendered at2x, printed242.
    readings = {1: ('boká', 'head'), 2: ('loa', 'iron'), 4: ('merom', 'goat'),
                7: ('itib', 'belly'), 8: ('suna', 'gold'), 9: ('hoi', 'yes'),
                12: ('hindrá', 'white'), 19: ('petah', 'female sex marker'),
                0: ('lanká', 'far'), 3: ('i-gota', 'three'), 5: ('itihi', 'hand'),
                6: ('lilai', 'cat'), 10: ('ondo', 'go'), 14: ('gandá-ni', 'four'),
                15: ('boy-i', 'mother'), 16: ('mándu', 'why'),
                17: ('Tumba', 'tree whose bark was used for clothing'), 18: ('goné', 'tooth')}
    findings = []
    for index, (form, gloss) in readings.items():
        unit = packet['sample_units'][index]
        actual = unit['actual_rows']
        assert len(actual) == 1 and actual[0][2:4] == [form, gloss]
        findings.append({'selection_index': index, 'key': unit['source_unit_key'],
                         'pdf_page': unit['pdf_page'], 'printed_page': unit['printed_page'],
                         'reading': form, 'gloss': gloss, 'result': 'pass'})
    for index, forms, gloss in [(11, ['árá', 'nái'], 'he'), (13, ['kano', 'kan-da'], 'son')]:
        unit = packet['sample_units'][index]
        actual = unit['actual_rows']
        assert [r[2] for r in actual] == forms and all(r[3] == gloss for r in actual)
        assert actual[1][11] == actual[0][10]
        findings.append({'selection_index': index, 'key': unit['source_unit_key'],
                         'pdf_page': unit['pdf_page'], 'printed_page': unit['printed_page'],
                         'readings': forms, 'gloss': gloss, 'result': 'pass',
                         'variant_review': 'Two printed responses for the same prompt; source-local variant target resolves.'})
    assert len(findings) == 20
    for index in [14, 16, 17]:
        actual = packet['sample_units'][index]['actual_rows'][0]
        assert 'uncertain' in actual[14].split() and actual[6]
    expected_markers = {'loko': ('plural marker', 'pl'), 'andesh': ('male sex marker', 'm'),
                        'petah': ('female sex marker', 'f')}
    for unit in packet['additional_morphology_edges']:
        actual = unit['actual_rows'][0]
        gloss, tag = expected_markers[actual[2]]
        assert actual[3] == gloss and tag in actual[14].split()
    report = {'status': 'passed independent frozen-output audit', 'reviewer': 'root',
              'manifest_sha256': sha(manifest_path), 'all_frozen_hashes_verified': True,
              'all_selected_rows_match_actual_proposal': True,
              'sample_units_reviewed': len(findings), 'sample_units_total': 20,
              'findings': sorted(findings, key=lambda f: f['selection_index']),
              'material_errors': 0, 'sample_rows_reviewed': 22,
              'additional_markers_reviewed': list(expected_markers),
              'marker_evidence': 'Printed242 explicitly states plural by adding loko and sex by adding andesh/petah; goat examples confirm respective masculine/feminine pairing. No affix spelling inferred.',
              'controls_reviewed': [
                  {'key': u['source_unit_key'], 'pdf_page': u['pdf_page'], 'result': 'pass',
                   'reason': 'Original Juanga column has ellipsis; no row emitted.' if u['status'] == 'printed-blank' else
                   'English narrator botanical/clothing label lacks explicit Juang lexical attribution; exclusion retained.'}
                  for u in packet['controls']],
              'method': 'Root read original PDF167,170,171,250,251,252,253,254,256,257 at2x, with6x crop for cramped numeral four. Compared complete selected rows, source grouping, locators, tags, uncertainty and variant fields against the pinned actual proposal.',
              'uncertainty_review': 'Four retains preferred n with m/n caveat; why retains source question-mark evidence in Notes and uncertain tag; Tumba remains attribution-uncertain with contextual definition and no species claim.',
              'remaining': 'Source-stage installation and shared bibliography reconciliation pending. Full CLDF/identity/graph/full-suite/browser gates remain deferred. This audit alone does not establish full ingestion completion.'}
    (O / 'root-independent-audit-progress-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (O / 'root-independent-whole-output-audit-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'all_frozen_hashes_verified', 'sample_units_reviewed']}))


if __name__ == '__main__':
    main()
