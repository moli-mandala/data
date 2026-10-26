"""Isolated four-target transaction tests; never execute canonical installation."""
import csv,importlib.util,json,shutil
from pathlib import Path
import pytest
SOURCE=Path(__file__).resolve().parents[2]/'import_source.py'
@pytest.fixture
def transaction(tmp_path):
    spec=importlib.util.spec_from_file_location('aranadan_installer',SOURCE);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    data=tmp_path/'data';here=data/'data/other/forms/raw_data/fixture';here.mkdir(parents=True);(data/'cldf').mkdir()
    (data/'data/form-identities.csv').write_text('unchanged durable identity')
    (data/'cldf/sources.bib').write_text('@misc{unrelated,title={Keep exact},year={1999}}\n')
    package=SOURCE.parent
    for role,name in mod.INPUTS.items():
        dst=here/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if role=='audit':
            rows=[json.loads(l) for l in (package/'integration/review-audit.jsonl').read_text().splitlines()]
            for r in rows:
                r['accepted_for_installation']=True
                r['entry_keys']=r['review_candidate_keys']
            dst.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        else:shutil.copyfile(package/name,dst)
    shutil.copyfile(SOURCE,here/'import_source.py');mod.DATA=data;mod.HERE=here;mod.APPROVAL=here/'root-installation-approval.json'
    targets=[data/f'data/other/forms/{mod.STEM}.csv',data/f'data/other/forms/{mod.STEM}.yaml',here/'installed-audit.jsonl',data/'cldf/sources.bib']
    approval=dict(installation_approved=True,result='pass',rows=102,audit_units=97,files={n:mod.file_hash(here/n) for n in [*mod.INPUTS.values(),'import_source.py']},canonical_baselines={str(p.relative_to(data)):mod.file_hash(p) for p in targets},durable_identity_sha256=mod.file_hash(data/'data/form-identities.csv'))
    mod.APPROVAL.write_text(json.dumps(approval));return mod,approval

def test_dry_run_preserves_canonical_bytes(transaction):
    mod,_=transaction
    before={str(p):p.read_bytes() for p in mod.DATA.rglob('*') if p.is_file()}
    mod.main()
    after={str(p):p.read_bytes() for p in mod.DATA.rglob('*') if p.is_file()}
    assert before==after

def test_input_and_identity_drift_stop_before_writes(transaction):
    mod,_=transaction
    source=mod.HERE/mod.INPUTS['csv']
    original=source.read_bytes()
    source.write_bytes(original+b'changed')
    with pytest.raises(AssertionError):mod.prepare()
    source.write_bytes(original)
    (mod.DATA/'data/form-identities.csv').write_text('changed identity')
    with pytest.raises(AssertionError):mod.prepare()

def simulated_journal(mod,planned):
    stage_dir=mod.HERE/'installation-stage'
    stage_dir.mkdir()
    targets={}
    for n,(target,content) in enumerate(planned.items()):
        stage=stage_dir/f'{n:02d}.asset'
        stage.write_bytes(content)
        targets[str(target.relative_to(mod.DATA))]=dict(staged_file=str(stage.relative_to(mod.HERE)),output_sha256=mod.digest(content))
    journal=dict(approval_sha256=mod.file_hash(mod.APPROVAL),targets=targets)
    (mod.HERE/'installation-transaction.json').write_text(json.dumps(journal))
    return journal

@pytest.mark.parametrize('written_count',range(1,5))
def test_simulated_partial_write_recovery_dry_run(transaction,written_count):
    mod,_=transaction
    _,planned,_=mod.prepare()
    simulated_journal(mod,planned)
    for target,content in list(planned.items())[:written_count]:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)  # Test fixture only; emulate interruption.
    before={str(p):p.read_bytes() for p in mod.DATA.rglob('*') if p.is_file()}
    mod.main()
    assert before=={str(p):p.read_bytes() for p in mod.DATA.rglob('*') if p.is_file()}

def test_complete_receipt_dry_run_is_idempotent(transaction):
    mod,_=transaction
    _,planned,identity=mod.prepare()
    for path,content in planned.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)  # Fixture state, not installer execution.
    receipt=dict(approval_sha256=mod.file_hash(mod.APPROVAL),durable_identity_sha256=mod.file_hash(identity),
                 installed_sha256={str(p.relative_to(mod.DATA)):mod.digest(b) for p,b in planned.items()})
    (mod.HERE/'installation-receipt.json').write_text(json.dumps(receipt))
    before={str(p):p.read_bytes() for p in mod.DATA.rglob('*') if p.is_file()}
    mod.main()
    assert before=={str(p):p.read_bytes() for p in mod.DATA.rglob('*') if p.is_file()}

def test_post_write_drift_prevents_receipt(transaction, monkeypatch):
    mod,_=transaction
    _,planned,_=mod.prepare()
    first,last=list(planned)[0],list(planned)[-1]
    original_replace=Path.replace
    def drift(path,target):
        result=original_replace(path,target)
        if Path(target)==last:
            first.write_text('Concurrent fixture edit')
        return result
    monkeypatch.setattr(Path,'replace',drift)
    with pytest.raises(AssertionError,match='Post-write target drift'):
        mod.main(install=True)
    assert not (mod.HERE/'installation-receipt.json').exists()
    assert (mod.HERE/'installation-transaction.json').exists()

def test_incomplete_receipt_rejects_missing_output_coverage(transaction):
    mod,_=transaction
    receipt=dict(approval_sha256=mod.file_hash(mod.APPROVAL),durable_identity_sha256=mod.file_hash(mod.DATA/'data/form-identities.csv'),installed_sha256={})
    (mod.HERE/'installation-receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):mod.main()
    assert not (mod.DATA/f'data/other/forms/{mod.STEM}.csv').exists()

def test_recovery_rejects_unexpected_canonical_edit(transaction):
    mod,_=transaction
    _,planned,_=mod.prepare()
    simulated_journal(mod,planned)
    (mod.DATA/'cldf/sources.bib').write_text('unrelated concurrent edit')
    with pytest.raises(AssertionError,match='Unexpected partial-transaction edit'):mod.main()

def test_recovery_rejects_staged_byte_drift(transaction):
    mod,_=transaction
    _,planned,_=mod.prepare()
    simulated_journal(mod,planned)
    (mod.HERE/'installation-stage/00.asset').write_text('corrupt')
    with pytest.raises(AssertionError):mod.main()

def test_interrupted_receipt_temp_does_not_block_validated_recovery(transaction):
    mod,_=transaction
    _,planned,_=mod.prepare()
    simulated_journal(mod,planned)
    for path,content in planned.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
    temporary=mod.HERE/'installation-receipt.tmp'
    temporary.write_text('{"status":')  # Interrupted final receipt write in fixture.
    before={str(p):p.read_bytes() for p in mod.DATA.rglob('*') if p.is_file()}
    mod.main()
    assert before=={str(p):p.read_bytes() for p in mod.DATA.rglob('*') if p.is_file()}
    assert not (mod.HERE/'installation-receipt.json').exists()

def test_recovery_rejects_missing_target(transaction):
    mod,_=transaction
    _,planned,_=mod.prepare()
    journal=simulated_journal(mod,planned)
    journal['targets'].pop(next(iter(journal['targets'])))
    (mod.HERE/'installation-transaction.json').write_text(json.dumps(journal))
    with pytest.raises(AssertionError):mod.main()

@pytest.mark.parametrize('mutation',['duplicate-key','broken-variant','pua','native','row-count'])
def test_reapproved_malformed_csv_is_rejected(transaction,mutation):
    mod,a=transaction;p=mod.HERE/mod.INPUTS['csv'];r=list(csv.reader(p.open()))
    if mutation=='duplicate-key':r[1][10]=r[0][10]
    elif mutation=='broken-variant':r[0][11]='missing'
    elif mutation=='pua':r[0][2]='\ue000'
    elif mutation=='native':r[0][4]='invented'
    else:r.pop()
    with p.open('w',newline='') as f:csv.writer(f).writerows(r)
    a['files'][mod.INPUTS['csv']]=mod.file_hash(p);mod.APPROVAL.write_text(json.dumps(a))
    with pytest.raises(AssertionError):mod.prepare()

@pytest.mark.parametrize('mutation',['missing-key','not-accepted','duplicate-unit'])
def test_reapproved_malformed_audit_is_rejected(transaction,mutation):
    mod,a=transaction;p=mod.HERE/mod.INPUTS['audit'];r=[json.loads(l) for l in p.read_text().splitlines()]
    if mutation=='missing-key':next(x for x in r if x['entry_keys'])['entry_keys']=[]
    elif mutation=='not-accepted':r[0]['accepted_for_installation']=False
    else:r[1]['source_unit_key']=r[0]['source_unit_key']
    p.write_text(''.join(json.dumps(x)+'\n' for x in r));a['files'][mod.INPUTS['audit']]=mod.file_hash(p);mod.APPROVAL.write_text(json.dumps(a))
    with pytest.raises(AssertionError):mod.prepare()
