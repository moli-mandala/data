"""Install only the independently approved frozen Birhor source stage. No DB build."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'whole-snapshot-recovery'
DATA = ROOT.parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(require_approval=False):
    manifest_path = OUT / 'frozen-manifest-20260927.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in {**manifest['files'], **manifest['review_inputs']}.items():
        if Path(name).name != name or digest(OUT / name) != expected:
            raise ValueError(f'Stale frozen source file: {name}')
    if digest(ROOT / 'prepare_full_source.py') != manifest['builder_sha256']:
        raise ValueError('Builder changed after source freeze')
    if digest(ROOT / 'lexical-snapshot.jsonl') != manifest['snapshot_sha256']:
        raise ValueError('Pinned source snapshot changed')
    if digest(DATA / manifest['focused_tests']['test_file']) != manifest['focused_tests']['test_sha256']:
        raise ValueError('Focused tests changed after source freeze')
    rows = list(csv.reader((OUT / 'proposal.csv').open()))
    audit = [json.loads(line) for line in (OUT / 'proposal-audit.jsonl').open()]
    assert len(rows) == manifest['rows'] == 3480
    keys = {r[10] for r in rows}
    assert len(keys) == len(rows)
    assert all(len(r) == 15 for r in rows)
    assert {k for a in audit for k in a['entry_keys']} == keys
    assert all(not r[11] or r[11] in keys and r[11] != r[10] for r in rows)
    assert sum(a['section'] == 'structured-head-and-senses' for a in audit) == 2746
    if require_approval:
        approval_path = OUT / 'root-installation-approval-20260927.json'
        if not approval_path.exists():
            raise ValueError('Final installation approval is absent')
        approval = json.loads(approval_path.read_text())
        if (approval.get('status') != 'approved' or
                approval.get('frozen_manifest_sha256') != digest(manifest_path)):
            raise ValueError('Approval does not certify this exact frozen proposal')
    return rows, audit, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    rows, audit, manifest_path = validate(args.install)
    if args.install:
        targets = [
            ('proposal.csv', DATA / 'data/other/forms/20260927-birhor-living.csv'),
            ('20260927-birhor-living.yaml', DATA / 'data/other/forms/20260927-birhor-living.yaml'),
            ('proposal-profile.txt', DATA / 'conversion/birhor-living.txt'),
            ('proposal-audit.jsonl', OUT / 'installed-audit.jsonl'),
        ]
        for name, target in targets:
            if target.exists() and digest(target) != digest(OUT / name):
                raise ValueError(f'Existing different canonical file: {target}')
        for name, target in targets:
            shutil.copyfile(OUT / name, target)
        report = {'status': 'source inputs installed; database and browser gates deferred',
                  'rows': len(rows), 'audit_units': len(audit),
                  'frozen_manifest_sha256': digest(manifest_path),
                  'files': {str(target.relative_to(DATA)): digest(target) for _, target in targets},
                  'bibliography': 'Exact three-entry proposal-sources.bib registration coordinated separately with root.'}
        (OUT / 'installation-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(rows)} rows and {len(audit)} audit units verified; no database build')


if __name__ == '__main__':
    main()
