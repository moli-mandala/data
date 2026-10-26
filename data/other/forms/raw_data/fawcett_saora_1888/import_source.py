"""Validate or install the approved whole Fawcett source stage; never build a DB."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[4]
APPROVAL_NAME = 'root-installation-approval-20260927.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(require_approval=False):
    manifest_path = P / 'frozen-manifest-20260927.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['hashes'].items():
        if Path(name).name != name or digest(P / name) != expected:
            raise ValueError(f'Stale or invalid frozen input: {name}')
    if digest(DATA / manifest['focused_tests']['path']) != manifest['focused_tests']['sha256']:
        raise ValueError('Focused tests changed after the freeze')
    rows = list(csv.reader((P / 'proposal.csv').open()))
    audit = [json.loads(line) for line in (P / 'proposal-audit.jsonl').read_text().splitlines()]
    assert len(rows) == manifest['row_count'] == 128
    assert len(audit) == manifest['unit_count'] == 164
    keys = {r[10] for r in rows}
    assert len(keys) == len(rows)
    assert {k for unit in audit for k in unit['entry_keys']} == keys
    assert all(not r[11] or r[11] in keys for r in rows)
    sample = json.loads((P / 'root-whole-output-audit-20260927-pass1.json').read_text())
    assert sample['material_errors'] == 0 and sample['source_units_reviewed'] == 20
    for name in ['proposal.csv', 'proposal-audit.jsonl', 'full-reviewed-inventory.jsonl']:
        assert digest(P / name) == sample['input_sha256'][name], name
    if require_approval:
        approval_path = P / APPROVAL_NAME
        if not approval_path.exists():
            raise ValueError('Final whole-source installation approval has not been recorded')
        approval = json.loads(approval_path.read_text())
        if (approval.get('status') != 'approved'
                or approval.get('frozen_manifest_sha256') != digest(manifest_path)):
            raise ValueError('Final approval does not match this frozen manifest')
    return rows, audit, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    rows, audit, manifest = validate(require_approval=args.install)
    if args.install:
        destinations = [
            ('proposal.csv', DATA / 'data/other/forms/20260927-fawcett-saora.csv'),
            ('proposal-audit.jsonl', P / 'audit.jsonl'),
            ('proposal-profile.txt', DATA / 'conversion/fawcett-saora-1888.txt'),
            ('20260927-fawcett-saora.yaml', DATA / 'data/other/forms/20260927-fawcett-saora.yaml'),
        ]
        # Validate every target before the first copy. Existing different data is
        # not overwritten by this first-install importer.
        for source, target in destinations:
            if target.exists() and digest(target) != digest(P / source):
                raise ValueError(f'Conflicting existing installation: {target}')
        for source, target in destinations:
            shutil.copyfile(P / source, target)
        installed = dict(status='source inputs installed; full build/browser gates deferred',
                         rows=len(rows), units=len(audit),
                         frozen_manifest_sha256=digest(P / 'frozen-manifest-20260927.json'),
                         files={str(target.relative_to(DATA)):digest(target) for _, target in destinations},
                         bibliography='Shared cldf/sources.bib registration is coordinated separately; reference-proposal.bib contains the exact entry.')
        (P / 'installation-manifest.json').write_text(json.dumps(installed, indent=2)+'\n')
    print(f'{len(rows)} source rows and {len(audit)} audited units verified; no database build')


if __name__ == '__main__':
    main()
