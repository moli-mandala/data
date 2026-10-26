"""Run only with a separately pinned root/independent correction approval."""
import argparse
import hashlib
import json
from pathlib import Path

from atomic_conservation_verifier import validate_atomic_accounting
from revised_boundary_verifier import apply_pinned_approved_corrections, validate_partition

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DERIVATIVE = BASE / 'specimens/sharp-reviewed-revision'
LEDGER_PIN = 'b720a686b0352f6205a9374fd7eec4c2ae370e02a581e41a483a3562bc08873e'
TYPED_KEYS = {
    'grierson1906kolamibasim:p564:specimen75-line:16:word:1',
    'grierson1906kolamibasim:p565:specimen75-line:7:word:6',
    'grierson1906kolamibasim:p569:specimen76-line:9:word:7',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--approval-sha256', required=True)
    args = parser.parse_args()
    assert sha(args.approval) == args.approval_sha256, 'Independent approval drift'
    approval = json.loads(args.approval.read_text())
    approved = approval if isinstance(approval, list) else approval.get('corrections', approval.get('cases'))
    assert isinstance(approved, list), 'Missing independently approved literal cases'
    snapshot = json.loads((HERE / 'sharp-derivative-preflight.json').read_text())
    for name, pin in snapshot['sha256'].items():
        assert sha(BASE / name) == pin, ('Derivative changed after snapshot', name)
    historical = json.loads((HERE / 'structural-conservation-review.json').read_text())
    physical_lines = []
    for page in [564, 565, 568, 569]:
        name = f'p{page}-first-reading.jsonl'
        path = BASE / 'specimens' / name
        assert sha(path) == historical['input_sha256'][name], ('Immutable reading drift', name)
        physical_lines.extend(lines(path))
    assert len(physical_lines) == 64
    immutable = [atom for line in physical_lines for atom in line['target_atoms']]
    assert len(immutable) == 561
    reviewed = apply_pinned_approved_corrections(immutable, DERIVATIVE / 'keyed-literal-overlay.json', LEDGER_PIN, approved, metadata_field='native_review_changes')
    actual = lines(DERIVATIVE / 'reviewed-physical-atoms.jsonl')
    assert len(actual) == len(reviewed)
    retained_uncertainties = {}
    for expected, observed in zip(reviewed, actual):
        copy = dict(observed)
        uncertainty = copy.pop('typed_uncertainties', None)
        assert copy == expected, ('Unapproved reviewed atom mutation', expected['source_unit_key'])
        if uncertainty is not None:
            key = expected['source_unit_key']
            assert key in TYPED_KEYS and uncertainty
            assert all(u['type'] == 'source-glyph-identity' and u['reason'] for u in uncertainty)
            expected['typed_uncertainties'] = uncertainty
            retained_uncertainties[key] = uncertainty
    assert set(retained_uncertainties) == TYPED_KEYS
    atomic = validate_atomic_accounting(immutable, reviewed, lines(DERIVATIVE / 'atomic-emission-proposal.jsonl'), lines(DERIVATIVE / 'physical-atom-emission-audit.jsonl'))
    boundary = validate_partition(reviewed, lines(DERIVATIVE / 'expression-emission-proposal.jsonl'), historical['expression_partitions'])
    old_partitions = {p['key']: p['physical_keys'] for p in historical['expression_partitions']}
    old_by_atom = {k: p['key'] for p in historical['expression_partitions'] for k in p['physical_keys']}
    expected_history = []
    for current in boundary['expression_partitions']:
        if old_partitions.get(current['key']) != current['physical_keys']:
            anchors = list(dict.fromkeys(old_by_atom[k] for k in current['physical_keys']))
            expected_history.append({'new_anchor': current['key'], 'new_contributors': current['physical_keys'], 'historical_anchors': anchors, 'historical_contributors': [old_partitions[k] for k in anchors]})
    assert json.loads((DERIVATIVE / 'expression-boundary-history.json').read_text()) == expected_history, 'Owner merge/split history omits or changes a contributor/anchor'
    report = {
        'reviewer': '/root/asur_finish',
        'status': 'passed_independently_approved_literal_overlay_atomic_and_boundary_conservation',
        'approval_path': str(args.approval), 'approval_sha256': args.approval_sha256,
        'correction_ledger_sha256': LEDGER_PIN, 'input_sha256': snapshot['sha256'],
        'physical_lines': len(physical_lines), 'atomic_accounting': atomic,
        'boundary_accounting': boundary, 'retained_typed_uncertainties': retained_uncertainties,
        'verified_changed_span_history': expected_history,
        'limits': ['Exact source accounting and approved-literal continuity; not a substitute for final combined CSV metadata/profile/reference/parser and fresh original-output audit.', 'Physical lines are not treated as sentences; English remains the exact aligned source join.', 'Merge/split anchor history is not a Form-ID alias or Variant relation.'],
        'canonical_mutations': False,
    }
    target = HERE / 'sharp-revised-conservation-review.json'
    assert not target.exists(), 'Preserve prior review; use a new revision instead of overwriting'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(target), 'sha256': sha(target), 'counts': boundary['counts']}))


if __name__ == '__main__':
    main()
