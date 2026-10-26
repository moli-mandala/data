"""Recoverable source-local literal correction; disabled until root activates plan."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
ROOT=DATA.parent

def file_hash(path):
    if not path.exists():return None
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def contained(root,name):
    path=(root/name).resolve();assert path.is_relative_to(root.resolve()),name
    return path

def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def validate(draft=False):
    path=HERE/('root-acceptance-draft.json' if draft else 'root-acceptance.json')
    plan=json.loads(path.read_text())
    assert plan['installation_approved'] is (False if draft else True)
    if not draft:assert plan['result']=='pass'
    assert set(plan['targets'])=={'csv','audit','importer'}
    for name,pin in plan['files'].items():assert file_hash(contained(HERE,name))==pin,name
    for asset in plan['targets'].values():
        assert asset['input'] in plan['files']
        assert asset['output_sha256']==plan['files'][asset['input']]
        contained(DATA,asset['target'])
    historical=PACKAGE/'root-installation-approval.json'
    assert file_hash(historical)==plan['root_historical_approval_sha256']
    for name,pin in json.loads(historical.read_text())['files'].items():
        archived=HERE/'historical-import_source.py' if name=='import_source.py' else contained(PACKAGE,name)
        assert file_hash(archived)==pin,('Historical approved input drift',name)
    evidence=plan['source_evidence']
    assert file_hash(contained(ROOT,evidence['path']))==evidence['sha256']
    return path,plan

def output_hashes(plan):
    return {role:file_hash(contained(DATA,asset['target'])) for role,asset in plan['targets'].items()}

def main(install=False,draft=False,rollback=False):
    assert not draft or not install
    approval,plan=validate(draft)
    receipt_path=HERE/'correction-receipt.json'
    journal_path=HERE/'correction-transaction.json'
    expected={role:a['output_sha256'] for role,a in plan['targets'].items()}
    if receipt_path.exists() and not rollback:
        receipt=json.loads(receipt_path.read_text())
        assert receipt['approval_sha256']==file_hash(approval)
        assert set(receipt['output_sha256'])==set(expected)
        assert receipt['output_sha256']==output_hashes(plan)==expected
        # Only own immutable settings/profile are permanent byte guards. Later
        # unrelated Bib/registry additions and legitimate identity builds are allowed.
        for name,pin in plan['unchanged_at_initial_transaction'].items():
            if name.endswith(('.yaml','grierson-kui-1906.txt')):assert file_hash(DATA/name)==pin,name
        print(json.dumps({'status':'already corrected; exact own assets verified','rows':1832}));return
    if journal_path.exists():
        journal=json.loads(journal_path.read_text())
        assert journal['approval_sha256']==file_hash(approval)
        assert set(journal['targets'])==set(plan['targets'])
        for role,asset in plan['targets'].items():
            record=journal['targets'][role]
            assert record['baseline_sha256']==asset['baseline_sha256'] and record['output_sha256']==asset['output_sha256']
            assert file_hash(contained(HERE,record['backup']))==asset['baseline_sha256']
            assert file_hash(DATA/asset['target']) in {asset['baseline_sha256'],asset['output_sha256']},role
    else:
        assert not rollback,'No interrupted transaction exists'
        for role,asset in plan['targets'].items():assert file_hash(DATA/asset['target'])==asset['baseline_sha256'],role
        journal=None
    # Initial and resumed transactions protect all unchanged canonical assets.
    for name,pin in plan['unchanged_at_initial_transaction'].items():assert file_hash(DATA/name)==pin,name
    identity=DATA/'data/form-identities.csv'
    assert file_hash(identity)==plan['durable_identity_sha256']
    if not install:
        print(json.dumps({'status':'disabled draft validated' if draft else 'approved correction dry-run validated',
            'planned_output_sha256':expected,'rows':1832,'other_raw_rows_unchanged':1831}));return
    assert not draft
    if journal is None:
        backups=HERE/'transaction-backups';backups.mkdir(exist_ok=True)
        journal={'approval_sha256':file_hash(approval),'targets':{}}
        for role,asset in plan['targets'].items():
            backup=backups/(role+'.original')
            if backup.exists():assert file_hash(backup)==asset['baseline_sha256']
            else:backup.write_bytes((DATA/asset['target']).read_bytes())
            journal['targets'][role]=dict(backup=str(backup.relative_to(HERE)),
                baseline_sha256=asset['baseline_sha256'],output_sha256=asset['output_sha256'])
        atomic_json(journal_path,journal)
    for role,asset in plan['targets'].items():
        target=contained(DATA,asset['target'])
        source=contained(HERE,journal['targets'][role]['backup'] if rollback else asset['input'])
        wanted=asset['baseline_sha256'] if rollback else asset['output_sha256']
        assert file_hash(target) in {asset['baseline_sha256'],asset['output_sha256']},role
        if file_hash(target)!=wanted:
            temporary=target.with_name(target.name+'.printed-amd-tmp')
            temporary.write_bytes(source.read_bytes());assert file_hash(temporary)==wanted
            temporary.replace(target)
    assert file_hash(identity)==plan['durable_identity_sha256']
    for name,pin in plan['unchanged_at_initial_transaction'].items():assert file_hash(DATA/name)==pin,name
    wanted={role:(a['baseline_sha256'] if rollback else a['output_sha256']) for role,a in plan['targets'].items()}
    assert output_hashes(plan)==wanted,'Final exact output guard failed; retain journal for recovery'
    if rollback:
        atomic_json(HERE/'rollback-receipt.json',dict(approval_sha256=file_hash(approval),output_sha256=wanted))
        if receipt_path.exists():receipt_path.unlink()
        print(json.dumps({'status':'exact original three targets restored; journal retained'}));return
    atomic_json(receipt_path,dict(approval_sha256=file_hash(approval),output_sha256=wanted,
        durable_identity_sha256=plan['durable_identity_sha256'],rows=1832,changed_fields=plan['canonical_field_changes']))
    print(json.dumps({'status':'source-stage literal corrected; no full build','output_sha256':wanted}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--install',action='store_true')
    p.add_argument('--check-draft',action='store_true');p.add_argument('--rollback',action='store_true')
    a=p.parse_args();main(install=a.install,draft=a.check_draft,rollback=a.rollback)
