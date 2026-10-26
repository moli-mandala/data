"""Isolated fixture checks only; write cycles use pytest temporary DATA roots.

Never execute the installer CLI --install against the real repository.
"""
import csv
import importlib.util
import io
import json
from pathlib import Path
import pytest
import yaml

SCRIPT=Path(__file__).with_name('install_source.py')
def csv_bytes(rows):
    s=io.StringIO(newline='');csv.writer(s).writerows(rows);return s.getvalue().encode()

@pytest.fixture
def tx(tmp_path):
    spec=importlib.util.spec_from_file_location('behera_install_test',SCRIPT)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    data=tmp_path/'data';here=data/'data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-recovery'
    (here/'installation').mkdir(parents=True);(here/'approved').mkdir()
    m.DATA=data;m.HERE=here;m.APPROVAL=here/'installation/root-acceptance.json'
    (here/'installation/install_source.py').write_bytes(SCRIPT.read_bytes())
    (data/'data/form-identities.csv').write_text('key,id\nold,durable\n')
    row=['kw','','a','word','','','','behera2022korwakodaku[p. 1]','','','old','','','','']
    extra=['Sadri','','i','I','','','','behera2022korwakodaku[p. 95]','','','new','','','','pron 1sg']
    old_yaml={'file':m.OLD+'.csv','sources':{m.SOURCE:{'transcription':{'profile':'sil-korwa-kodaku'},'forms':{'split_alternates':False},'notes':{'audit_only':True}}}}
    changed_yaml={'file':m.OLD+'.csv','defaults':{'transcription':{'profile':'sil-korwa-kodaku'}},'sources':{m.SOURCE:{'forms':{'split_alternates':False},'notes':{'audit_only':True},'reference':{'editor':'OpenAI Codex','ocr':False,'etymology_provenance':'source'}}}}
    new_yaml={'file':m.NEW+'.csv','defaults':{'transcription':{'convert':False,'preserve_hyphens':True}}}
    changed_yaml['sources'][m.SOURCE]['identity']={'dedupe_by_entry_key':True}
    command=[str((here/'installation/install_source.py').relative_to(data)),'--install']
    for settings in (changed_yaml,new_yaml):settings['defaults']['importer']={'commands':[command]}
    old_bib='@misc{behera2022korwakodaku,title={Old scope}}\n@misc{unrelated,title={Keep exact text}}\n'
    new_bib='@misc{behera2022korwakodaku,title={Whole scope}}\n@misc{unrelated,title={Keep exact text}}\n@misc{aux,title={Reported only}}\n'
    baselines={'legacy_csv':csv_bytes([row]),'legacy_yaml':yaml.safe_dump(old_yaml).encode(),'languages':b'ID,Name\nkw,Korwa\n','dialects':b'ID,Name\nold,Old\n','bibliography':old_bib.encode()}
    finals={'legacy_csv':csv_bytes([row]),'addition_csv':csv_bytes([extra]),'legacy_yaml':yaml.safe_dump(changed_yaml).encode(),'addition_yaml':yaml.safe_dump(new_yaml).encode(),'languages':b'ID,Name\nkw,Korwa\nSadri,Sadri\n','dialects':b'ID,Name\nold,Old\nnew,New\n','bibliography':new_bib.encode(),'audit':''.join(json.dumps({'audit_key':k,'entry_keys':[k],'accepted_for_installation':True})+'\n' for k in ('old','new')).encode()}
    for role,b in baselines.items():
        p=data/m.TARGETS[role];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    assets={}
    for role,b in finals.items():
        name=f'approved/{role}.asset';(here/name).write_bytes(b)
        assets[role]={'input':name,'output_sha256':m.digest(b)}
    a={'installation_approved':True,'result':'pass','review_scope':'whole-report','engine_sha256':m.file_hash(m.ENGINE),'assets':assets,'files':{v['input']:v['output_sha256'] for v in assets.values()},'canonical_baselines':{n:m.file_hash(data/n) for n in m.TARGETS.values()},'durable_identity_sha256':m.file_hash(data/m.IDENTITY),'legacy_rows':1,'addition_rows':1,'rows':2,'audit_units':2,'legacy_changed_keys':[],'legacy_added_keys':[],'registry_added_ids':{'languages':['Sadri'],'dialects':['new']},'bibliography_changed_keys':[m.SOURCE],'bibliography_added_keys':['aux'],'audit_schema':{'key_field':'audit_key','entry_keys_field':'entry_keys'}}
    a['files']['installation/install_source.py']=m.file_hash(here/'installation/install_source.py')
    m.APPROVAL.write_text(json.dumps(a));return m,a

def snapshot(root):return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}

def simulate_journal(m,a,partial=0):
    _,plan,_=m.prepare();stage=m.HERE/'installation-stage';stage.mkdir()
    records={}
    for i,(path,content) in enumerate(plan.items()):
        p=stage/f'{i:02}.asset';p.write_bytes(content)
        records[str(path.relative_to(m.DATA))]={'staged_file':str(p.relative_to(m.HERE)),'output_sha256':m.digest(content)}
        if i<partial:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    (m.HERE/'installation-transaction.json').write_text(json.dumps({'approval_sha256':m.file_hash(m.APPROVAL),'targets':records}))
    return plan

def test_dry_run_is_read_only(tx):
    m,_=tx;before=snapshot(m.DATA);m.main();assert snapshot(m.DATA)==before

def test_missing_approval_refuses(tx):
    m,_=tx;m.APPROVAL.unlink();before=snapshot(m.DATA)
    with pytest.raises(FileNotFoundError):m.main()
    assert snapshot(m.DATA)==before

@pytest.mark.parametrize('which',['input','canonical','identity','engine'])
def test_drift_refused_before_writes(tx,which):
    m,a=tx
    if which=='input':(m.HERE/a['assets']['addition_csv']['input']).write_text('changed')
    elif which=='canonical':(m.DATA/m.TARGETS['legacy_csv']).write_text('changed')
    elif which=='identity':(m.DATA/m.IDENTITY).write_text('changed')
    else:a['engine_sha256']='0'*64;m.APPROVAL.write_text(json.dumps(a))
    before=snapshot(m.DATA)
    with pytest.raises(AssertionError):m.main()
    assert snapshot(m.DATA)==before

@pytest.mark.parametrize('partial',[1,4,8])
def test_interruption_recovery_dry_run(tx,partial):
    m,a=tx;simulate_journal(m,a,partial);before=snapshot(m.DATA)
    m.main();assert snapshot(m.DATA)==before

def test_journal_cannot_substitute_unapproved_output(tx):
    m,a=tx;simulate_journal(m,a)
    path=m.HERE/'installation-transaction.json';j=json.loads(path.read_text())
    name=next(iter(j['targets']));entry=j['targets'][name]
    (m.HERE/entry['staged_file']).write_text('evil');entry['output_sha256']=m.digest(b'evil')
    path.write_text(json.dumps(j))
    with pytest.raises(AssertionError,match='Unapproved recovery output'):m.main()

def test_repeat_complete_receipt_is_noop(tx):
    m,a=tx;plan=simulate_journal(m,a,8)
    (m.HERE/'installation-receipt.json').write_text(json.dumps({'approval_sha256':m.file_hash(m.APPROVAL),'durable_identity_sha256':a['durable_identity_sha256'],'installed_sha256':{str(p.relative_to(m.DATA)):m.digest(b) for p,b in plan.items()}}))
    before=snapshot(m.DATA);m.main();assert snapshot(m.DATA)==before

def test_unrelated_registry_and_bib_cannot_change(tx):
    m,a=tx
    for role in ['languages','bibliography']:
        asset=a['assets'][role];p=m.HERE/asset['input'];original=p.read_bytes()
        changed=original.replace(b'Korwa',b'Changed') if role=='languages' else original.replace(b'Keep exact text',b'Changed')
        p.write_bytes(changed);a['files'][asset['input']]=asset['output_sha256']=m.digest(changed);m.APPROVAL.write_text(json.dumps(a))
        with pytest.raises(AssertionError):m.prepare()
        p.write_bytes(original);a['files'][asset['input']]=asset['output_sha256']=m.digest(original);m.APPROVAL.write_text(json.dumps(a))

def test_rollback_plan_restores_exact_baselines_and_refuses_edits(tx):
    m,a=tx;m.preserve_baselines(a);simulate_journal(m,a,8)
    before=snapshot(m.DATA);m.main(rollback=True);assert snapshot(m.DATA)==before
    plan=m.rollback_plan(a)
    assert sum(b is None for b in plan.values())==3
    assert m.digest(plan[m.DATA/m.TARGETS['legacy_yaml']])==a['canonical_baselines'][m.TARGETS['legacy_yaml']]
    (m.DATA/m.TARGETS['bibliography']).write_text('concurrent edit')
    with pytest.raises(AssertionError,match='Rollback would overwrite unexpected edit'):m.rollback_plan(a)

def test_unknown_target_role_and_path_escape_refused(tx):
    m,a=tx;a['assets']['extra']=a['assets']['audit'];m.APPROVAL.write_text(json.dumps(a))
    with pytest.raises(AssertionError):m.main()
    with pytest.raises(AssertionError,match='Escaping path'):m.contained(m.DATA,'../outside')

def test_fixture_install_repeat_and_rollback(tx):
    m,a=tx
    original={name:(m.DATA/name).read_bytes() if (m.DATA/name).exists() else None for name in m.TARGETS.values()}
    assert str(m.DATA).endswith('/data') and 'test_fixture_install' in str(m.DATA)
    m.main(install=True)
    assert (m.HERE/'installation-receipt.json').exists()
    before=snapshot(m.DATA);m.main(install=True);assert snapshot(m.DATA)==before
    m.main(install=True,rollback=True)
    assert all(((m.DATA/name).read_bytes() if (m.DATA/name).exists() else None)==b for name,b in original.items())
    assert (m.HERE/'installation/rollback-receipt.json').exists()
    with pytest.raises(AssertionError,match='Rolled-back transaction'):m.main()

def test_fixture_partial_transaction_resumes(tx):
    m,a=tx;m.preserve_baselines(a);plan=simulate_journal(m,a,3)
    assert 'test_fixture_partial' in str(m.DATA)
    m.main(install=True)
    assert all(p.read_bytes()==b for p,b in plan.items())
    assert (m.HERE/'installation-receipt.json').exists()

def test_source_entry_identity_guard_required(tx):
    m,a=tx;asset=a['assets']['legacy_yaml'];p=m.HERE/asset['input']
    content=yaml.safe_load(p.read_text());del content['sources'][m.SOURCE]['identity']
    p.write_text(yaml.safe_dump(content));a['files'][asset['input']]=asset['output_sha256']=m.file_hash(p)
    m.APPROVAL.write_text(json.dumps(a))
    with pytest.raises((AssertionError,KeyError)):m.prepare()

def test_post_write_drift_prevents_success_receipt(tx, monkeypatch):
    m,a=tx
    original_replace=Path.replace
    last=m.DATA/m.TARGETS['audit']
    earlier=m.DATA/m.TARGETS['bibliography']
    def drift_after_last_write(path,target):
        result=original_replace(path,target)
        if Path(target)==last:
            earlier.write_text('Concurrent bibliography edit')
        return result
    monkeypatch.setattr(Path,'replace',drift_after_last_write)
    with pytest.raises(AssertionError,match='Post-write target drift'):
        m.main(install=True)
    assert not (m.HERE/'installation-receipt.json').exists()
    assert (m.HERE/'installation-transaction.json').exists()
