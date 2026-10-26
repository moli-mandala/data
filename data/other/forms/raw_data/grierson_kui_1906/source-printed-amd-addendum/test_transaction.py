"""Isolated guarded-transaction regressions; all temporary writes stay in Jambu."""
import importlib.util
import json
import tempfile
from pathlib import Path
import pytest

ACTUAL=Path(__file__).resolve().parent
WORKSPACE=next(p for p in ACTUAL.parents if (p/'data/source_meta.py').exists())

@pytest.fixture
def fixture(monkeypatch):
    spec=importlib.util.spec_from_file_location('kui_amd_fixture',ACTUAL/'apply_correction.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='kui-amd-fixture-',dir=WORKSPACE/'tmp') as name:
        root=Path(name);data=root/'data';package=data/'raw';here=package/'addendum'
        here.mkdir(parents=True);(data/'data').mkdir();(here/'stage').mkdir()
        monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'DATA',data)
        monkeypatch.setattr(module,'PACKAGE',package);monkeypatch.setattr(module,'HERE',here)
        (package/'root-installation-approval.json').write_text('{"files":{}}')
        evidence=root/'evidence.json';evidence.write_text('{}')
        (data/'data/form-identities.csv').write_text('stable identities')
        unchanged={}
        for filename in ['own.yaml','grierson-kui-1906.txt','sources.bib','languages.csv']:
            (data/filename).write_text('initial '+filename);unchanged[filename]=module.file_hash(data/filename)
        targets={};files={}
        for role in ['csv','audit','importer']:
            path=data/(role+'.own');path.write_text('old '+role)
            staged=here/'stage'/role;staged.write_text('new '+role)
            targets[role]=dict(target=path.name,input='stage/'+role,baseline_sha256=module.file_hash(path),output_sha256=module.file_hash(staged))
            files['stage/'+role]=module.file_hash(staged)
        plan=dict(installation_approved=True,result='pass',targets=targets,files=files,
            unchanged_at_initial_transaction=unchanged,durable_identity_sha256=module.file_hash(data/'data/form-identities.csv'),
            root_historical_approval_sha256=module.file_hash(package/'root-installation-approval.json'),
            source_evidence=dict(path='evidence.json',sha256=module.file_hash(evidence)),canonical_field_changes=[dict(column=2,old='and',new='amd')])
        (here/'root-acceptance.json').write_text(json.dumps(plan))
        draft=dict(plan,installation_approved=False,result='disabled')
        (here/'root-acceptance-draft.json').write_text(json.dumps(draft))
        yield module,plan

def test_dry_run_and_disabled_draft_write_nothing(fixture):
    m,p=fixture;before=m.output_hashes(p)
    m.main(draft=True);m.main();assert m.output_hashes(p)==before
    assert not (m.HERE/'correction-transaction.json').exists()

def test_disabled_approval_cannot_install(fixture):
    m,p=fixture;p['installation_approved']=False;(m.HERE/'root-acceptance.json').write_text(json.dumps(p))
    with pytest.raises(AssertionError):m.main(install=True)

def test_clean_install_noop_allows_later_shared_additions_and_identity_build(fixture):
    m,p=fixture;m.main(install=True)
    for name in ['sources.bib','languages.csv','data/form-identities.csv']:(m.DATA/name).write_text('legitimate later state')
    m.main(install=True)
    assert m.output_hashes(p)=={r:a['output_sha256'] for r,a in p['targets'].items()}

def test_unexpected_target_edit_stops_before_writes(fixture):
    m,p=fixture;(m.DATA/p['targets']['audit']['target']).write_text('unexpected')
    with pytest.raises(AssertionError):m.main(install=True)
    assert not (m.HERE/'correction-transaction.json').exists()

def interrupt_audit_write(m,monkeypatch):
    real=Path.replace
    def replacement(self,target):
        if self.name=='audit.own.printed-amd-tmp':raise RuntimeError('simulated interruption')
        return real(self,target)
    monkeypatch.setattr(Path,'replace',replacement)
    with pytest.raises(RuntimeError):m.main(install=True)
    monkeypatch.setattr(Path,'replace',real)

def test_interrupted_write_resumes_exact_journal(fixture,monkeypatch):
    m,p=fixture;interrupt_audit_write(m,monkeypatch)
    assert (m.HERE/'correction-transaction.json').exists() and not (m.HERE/'correction-receipt.json').exists()
    m.main(install=True)
    assert m.output_hashes(p)=={r:a['output_sha256'] for r,a in p['targets'].items()}

def test_interrupted_write_rolls_back_only_exact_targets(fixture,monkeypatch):
    m,p=fixture;interrupt_audit_write(m,monkeypatch);m.main(install=True,rollback=True)
    assert m.output_hashes(p)=={r:a['baseline_sha256'] for r,a in p['targets'].items()}
    assert not (m.HERE/'correction-receipt.json').exists()

def test_post_write_drift_never_gets_success_receipt(fixture,monkeypatch):
    m,p=fixture;real=Path.replace
    def replacement(self,target):
        result=real(self,target)
        if self.name=='importer.own.printed-amd-tmp':(m.DATA/'csv.own').write_text('raced output')
        return result
    monkeypatch.setattr(Path,'replace',replacement)
    with pytest.raises(AssertionError):m.main(install=True)
    assert not (m.HERE/'correction-receipt.json').exists()

def test_incomplete_receipt_is_rejected(fixture):
    m,p=fixture;m.main(install=True);path=m.HERE/'correction-receipt.json'
    receipt=json.loads(path.read_text());del receipt['output_sha256']['importer'];path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):m.main(install=True)
