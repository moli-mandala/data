"""Six-target transaction-only fixtures, entirely inside Jambu scratch.

These intentionally substitute miniature payload validation. They test transaction
safety, not whole-source asset acceptance; real asset/parser gates remain separate.
"""
import hashlib,importlib.util,json,tempfile
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
@pytest.fixture
def transaction(monkeypatch):
    pin=sha(HERE/'root-acceptance-draft.json')
    assert not (HERE/'root-acceptance.json').exists()
    with tempfile.TemporaryDirectory(dir=DATA.parent/'tmp/jambu-next-source') as name:
        workspace=Path(name);data=workspace/'data';package=data/'data/other/forms/raw_data/grierson_kolami_basim_1906';local=package/'installation';local.mkdir(parents=True)
        spec=importlib.util.spec_from_file_location('kolami_fixture_transaction',HERE/'install_source.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        monkeypatch.setattr(m,'HERE',local);monkeypatch.setattr(m,'PACKAGE',package);monkeypatch.setattr(m,'DATA',data);monkeypatch.setattr(m,'APPROVAL',local/'root-acceptance.json')
        (local/'install_source.py').write_bytes((HERE/'install_source.py').read_bytes())
        refs='@misc{grierson1906kolamibasim,title={Fixture primary}}\n@misc{haig1897gondi_kolami,title={Fixture authority}}\n'
        (local/'staged-references.bib').write_text(refs)
        dialects='ID,Language_ID,Name\nfixture_wun,Kolami,Fixture Wun\nfixture_basim,Naikri,Fixture Basim\n'
        (local/'staged-dialects.csv').write_text(dialects)
        payload={'csv':b'fixture csv\n','yaml':b'fixture yaml\n','audit':b'fixture audit\n','profile':b'fixture profile\n',
                 'bib':('@misc{previous,title={Old}}\n'+refs).encode(),'dialects':('ID,Language_ID,Name\nprevious,Kolami,Old\n'+dialects.split('\n',1)[1]).encode()}
        assets={};baselines={};files={'installation/install_source.py':sha(local/'install_source.py')}
        for role,target in m.TARGETS.items():
            stage=local/(role+'.asset');stage.write_bytes(payload[role]);rel=str(stage.relative_to(package));files[rel]=sha(stage)
            assets[role]={'target':target,'input':rel,'output_sha256':sha(stage)}
            p=data/target;p.parent.mkdir(parents=True,exist_ok=True)
            if role=='bib':p.write_text('@misc{previous,title={Old}}\n')
            if role=='dialects':p.write_text('ID,Language_ID,Name\nprevious,Kolami,Old\n')
            baselines[target]=sha(p)
        for n in ['staged-references.bib','staged-dialects.csv']:files['installation/'+n]=sha(local/n)
        for role,name in [('bib','baseline-bibliography.bib'),('dialects','baseline-dialects.csv')]:
            (local/name).write_bytes((data/m.TARGETS[role]).read_bytes());files['installation/'+name]=sha(local/name)
        identity=data/'data/form-identities.csv';identity.parent.mkdir(parents=True,exist_ok=True);identity.write_bytes(b'fixture identity')
        original=workspace/'tmp/original';original.parent.mkdir(parents=True);original.write_bytes(b'fixture original')
        a={'installation_approved':True,'result':'pass','shared_baseline_captured':True,'audit_acceptance_ready':True,
           'assets':assets,'canonical_baselines':baselines,'files':files,'original_inputs':{'tmp/original':sha(original)},
           'durable_identity_sha256':sha(identity),'canonical_reuse_inputs':{},'canonical_reuse_rows':[]}
        m.APPROVAL.write_text(json.dumps(a))
        def miniature_payload_validator(approval,outputs):
            assert set(outputs)==set(m.TARGETS) and outputs==payload
        monkeypatch.setattr(m,'validate_assets',miniature_payload_validator)
        yield m,a
    assert sha(HERE/'root-acceptance-draft.json')==pin and not (HERE/'root-acceptance.json').exists()
def test_six_targets_and_repeat_after_shared_additions(transaction):
    m,a=transaction;assert not m.main()['canonical_mutations'];r=m.main(install=True);assert len(r['installed_sha256'])==6
    with (m.DATA/m.TARGETS['bib']).open('a') as f:f.write('@misc{later,title={Later}}\n')
    with (m.DATA/m.TARGETS['dialects']).open('a') as f:f.write('later,Kolami,Later\n')
    (m.DATA/'data/form-identities.csv').write_bytes(b'legitimate later identity build')
    before={n:sha(m.DATA/n) for n in m.TARGETS.values()}
    assert not m.main(install=True)['canonical_mutations'];assert before=={n:sha(m.DATA/n) for n in m.TARGETS.values()}
def test_profile_own_byte_drift_is_rejected(transaction):
    m,a=transaction;m.main(install=True);(m.DATA/m.TARGETS['profile']).write_bytes(b'unexpected')
    with pytest.raises(AssertionError):m.main(install=True)
def test_shared_baseline_drift_prevents_transaction(transaction):
    m,a=transaction;(m.DATA/m.TARGETS['dialects']).write_bytes(b'unexpected')
    with pytest.raises(AssertionError,match='canonical baseline drift'):m.main(install=True)
    assert not (m.HERE/'installation-transaction.json').exists()
def test_interrupt_after_shared_write_recovers_exact_staged_outputs(transaction,monkeypatch):
    m,a=transaction;replace=Path.replace;state={'interrupted':False}
    def interrupted(path,target):
        r=replace(path,target)
        if Path(target)==m.DATA/m.TARGETS['bib'] and not state['interrupted']:
            state['interrupted']=True;raise OSError('fixture shared-write interruption')
        return r
    monkeypatch.setattr(Path,'replace',interrupted)
    with pytest.raises(OSError):m.main(install=True)
    assert not (m.HERE/'installation-receipt.json').exists()
    assert len(m.main(install=True)['installed_sha256'])==6
def test_receipt_interruption_resumes(transaction,monkeypatch):
    m,a=transaction;replace=Path.replace;state={'interrupted':False}
    def interrupted(path,target):
        if Path(target)==m.HERE/'installation-receipt.json' and not state['interrupted']:
            state['interrupted']=True;raise OSError('fixture receipt interruption')
        return replace(path,target)
    monkeypatch.setattr(Path,'replace',interrupted)
    with pytest.raises(OSError):m.main(install=True)
    assert len(m.main(install=True)['installed_sha256'])==6
def test_postwrite_drift_never_receipted_or_rolled_back(transaction,monkeypatch):
    m,a=transaction;replace=Path.replace
    def changed(path,target):
        r=replace(path,target)
        if Path(target)==m.DATA/m.TARGETS['dialects']:(m.DATA/m.TARGETS['csv']).write_bytes(b'external edit')
        return r
    monkeypatch.setattr(Path,'replace',changed)
    with pytest.raises(AssertionError,match='post-write drift'):m.main(install=True)
    assert not (m.HERE/'installation-receipt.json').exists()
    with pytest.raises(AssertionError,match='partial transaction drift'):m.main(install=True,rollback=True)
def test_rollback_restores_all_six_exact_baselines(transaction):
    m,a=transaction;m.main(install=True);m.main(install=True,rollback=True)
    assert all(sha(m.DATA/n)==pin for n,pin in a['canonical_baselines'].items())
def test_real_disabled_coordinator_cannot_install():
    spec=importlib.util.spec_from_file_location('kolami_disabled',HERE/'install_source.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with pytest.raises(FileNotFoundError):m.main(install=True)
    assert not (HERE/'installation-transaction.json').exists()
