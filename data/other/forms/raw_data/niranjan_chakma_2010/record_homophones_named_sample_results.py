"""Record original-page comparison of the fixed twenty-record root sample."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT / 'homophones-named-root-fresh-sample-20260927.json'
payload = json.loads(path.read_text())
holds = {
    'niranjanchakma2010:p66:homophones:left:g141:c1': {
        'field': 'form', 'old': 'পাট', 'candidate': 'পাঁট',
        'reason': 'Possible upper nasal mark in original PDF68 second-half render; needs enlarged comparison before accepting or rejecting the existing reading.'},
    'niranjanchakma2010:p72:named-groups:g48:c2': {
        'field': 'form', 'old': 'দোভঅ কাধি', 'candidate': None,
        'reason': 'Medial consonant shape in দোভঅ remains difficult to distinguish from ত at the reviewed PDF74 second-half resolution; retain exact first reading pending enlarged comparison.'},
}
assert len(payload['sample']) == 20
assert len({r['source_unit_key'] for r in payload['sample']}) == 20
assert set(holds) <= {r['source_unit_key'] for r in payload['sample']}
for row in payload['sample']:
    finding = holds.get(row['source_unit_key'])
    row['root_audit_status'] = 'typed transcription uncertainty' if finding else 'agrees with original page'
    row['root_review_method'] = 'Independent root comparison with original PDF page renders at 3x, including shared definitions and printed group attribution.'
    row['root_findings'] = [finding] if finding else []
    if finding:
        assert row['form'] == finding['old']
payload['result'] = {'reviewed': 20, 'agrees': 18, 'unresolved_transcription': 2,
                     'clean_audit_claim': False, 'canonical_changes': False,
                     'next_action': 'Inspect the two keyed readings at higher resolution; record resolutions and repeat fresh sampling if a material correction is established.'}
resolution_path = ROOT / 'h-font-control-review/two-root-sample-holds-review-20260927.json'
resolution = json.loads(resolution_path.read_text())
checks = {r['source_unit_key']: r for r in resolution['checks']}
assert set(checks) == set(holds)
for row in payload['sample']:
    if row['source_unit_key'] not in checks:
        continue
    check = checks[row['source_unit_key']]
    assert check['decision'] == 'retain' and check['reading'] == row['form']
    image = resolution_path.parent / check['evidence']
    assert hashlib.sha256(image.read_bytes()).hexdigest() == check['evidence_sha256']
    row['root_initial_findings'] = row['root_findings']
    row['root_findings'] = []
    row['root_audit_status'] = 'agrees with original page after enlarged control comparison'
    row['root_resolution'] = {**check, 'root_original_image_reviewed': True,
                              'review_note': 'Root independently viewed both 7x control crops and agrees with retained readings.'}
payload['result'] = {'reviewed': 20, 'agrees': 20, 'unresolved_transcription': 0,
                     'material_errors': 0, 'canonical_changes': False,
                     'scope': 'Fixed fresh sample from these two sections only; broader recorded uncertainties remain.',
                     'resolution_report_sha256': hashlib.sha256(resolution_path.read_bytes()).hexdigest()}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(payload['result']))
