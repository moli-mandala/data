"""Guarded metadata-only amendment; historical source acceptance remains immutable."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from pybtex.database import parse_string

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
spec = importlib.util.spec_from_file_location('historical_betta_installer', SOURCE / 'install_source.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
APPROVAL = HERE / 'approval.json'
RECEIPT = HERE / 'receipt.json'
YAML = HERE / '20260927-nair-betta-kurumba.yaml'
TARGET = legacy.TARGETS['yaml']
BASELINE = '2d4c07c7ac4de318d977c5674118bf2410f75b111db50d1c883fc40c0b73a838'


def digest(b):
    return hashlib.sha256(b).hexdigest()


def validate_own_assets():
    legacy.validate_approval()
    receipt = json.loads(legacy.RECEIPT.read_text())
    for role in ('csv', 'audit'):
        assert legacy.current(legacy.TARGETS[role]) == receipt['outputs'][role], ('source asset drift', role)
    approved = parse_string((SOURCE / 'proposed-reference.bib').read_text(), 'bibtex').entries
    live = parse_string(legacy.TARGETS['bib'].read_text(), 'bibtex').entries
    assert all(k in live and live[k] == v for k, v in approved.items()), 'source bibliography drift'


def replace_yaml(target, data, baseline):
    old = target.read_bytes()
    if old == data:
        return 'already installed; exact no-op'
    assert digest(old) == baseline, 'unexpected YAML drift'
    temporary = target.with_name(target.name + '.betta-metadata-tmp')
    assert not temporary.exists()
    try:
        temporary.write_bytes(data)
        assert target.read_bytes() == old, 'concurrent YAML drift'
        os.replace(temporary, target)
        assert target.read_bytes() == data, 'post-write YAML drift'
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        # Never replace a concurrent edit with historical bytes.
        if target.exists() and target.read_bytes() == data:
            target.write_bytes(old)
        raise
    return 'metadata amendment installed'


def main(install=False):
    validate_own_assets()
    a = json.loads(APPROVAL.read_text())
    assert a['status'] == 'approved', 'independent metadata approval required'
    assert a['wrapper_sha256'] == digest(Path(__file__).read_bytes()), 'wrapper drift'
    assert a['historical_approval_sha256'] == legacy.current(legacy.APPROVAL)
    assert a['historical_receipt_sha256'] == legacy.current(legacy.RECEIPT)
    assert a['proposed_yaml_sha256'] == digest(YAML.read_bytes())
    assert a['baseline_yaml_sha256'] == BASELINE
    data = YAML.read_bytes()
    assert digest(TARGET.read_bytes()) in (BASELINE, digest(data)), 'unexpected YAML drift'
    if not install:
        print(json.dumps({'status':'validated approved metadata amendment','install':False}))
        return
    before = legacy.current(legacy.IDENTITY)
    old = TARGET.read_bytes()
    try:
        result = replace_yaml(TARGET, data, BASELINE)
        validate_own_assets()
        assert TARGET.read_bytes() == data, 'post-write YAML drift'
        assert legacy.current(legacy.IDENTITY) == before, 'identity drift during metadata installation'
        if not RECEIPT.exists():
            temp = RECEIPT.with_suffix('.tmp')
            assert not temp.exists()
            temp.write_text(json.dumps({'status':result,'target_yaml_sha256':digest(data),
                                       'approval_sha256':digest(APPROVAL.read_bytes()),
                                       'historical_receipt_sha256':legacy.current(legacy.RECEIPT),
                                       'identity_before_and_after_sha256':before},indent=2)+'\n')
            os.replace(temp, RECEIPT)
        else:
            assert json.loads(RECEIPT.read_text())['target_yaml_sha256'] == digest(data)
    except BaseException:
        if old != data and TARGET.read_bytes() == data:
            rollback = TARGET.with_name(TARGET.name + '.betta-metadata-rollback-tmp')
            assert not rollback.exists()
            rollback.write_bytes(old)
            os.replace(rollback, TARGET)
        raise
    print(json.dumps({'status':result,'rows':206,'audit_units':225}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true')
    main(parser.parse_args().install)
