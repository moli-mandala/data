"""Tiny isolated actual-payload transaction fixtures; no canonical writes."""
import csv
import importlib.util
import io
import json
import tempfile
from pathlib import Path
import pytest
import yaml

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p/'source_meta.py').exists())

@pytest.fixture
def transaction(monkeypatch):
    with tempfile.TemporaryDirectory(dir=DATA.parent/'tmp') as folder:
        workspace=Path(folder); data=workspace/'data'
        package=data/'data/other/forms/raw_data/grierson_ahirwati_1908'
        local=package/'installation'; local.mkdir(parents=True)
        spec=importlib.util.spec_from_file_location('ahirwati_fixture',HERE/'install_source.py')
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        for key,value in [('HERE',local),('PACKAGE',package),('DATA',data),('APPROVAL',local/'root-acceptance.json')]:
            monkeypatch.setattr(m,key,value)
        (local/'install_source.py').write_bytes((HERE/'install_source.py').read_bytes())
        prior_bib='@misc{previous,title={Old unrelated source}}\n'
        refs='@misc{fixture,title={Approved source}}\n'
        prior_dialects='ID,Language_ID,Name\nprevious,Ahirwadi,Old\n'
        own_dialects='ID,Language_ID,Name\nfixture,Ahirwadi,Fixture\n'
        (local/'baseline-bibliography.bib').write_text(prior_bib)
        (local/'staged-references.bib').write_text(refs)
        (local/'baseline-dialects.csv').write_text(prior_dialects)
        (local/'staged-dialects.csv').write_text(own_dialects)
        settings={'file':m.STEM+'.csv','defaults':{
            'transcription':{'profile':'grierson-ahirwati-1908','input':'form','preserve_hyphens':True},
            'identity':{'dedupe_by_entry_key':True},'importer':{'commands':[['data/other/forms/raw_data/grierson_ahirwati_1908/installation/install_source.py','--install']]}},
            'sources':{'fixture':{'reference':{'editor':'OpenAI Codex','ocr':False,'etymology_provenance':'source'},
                                  'forms':{'split_alternates':False},'gloss':{'grammar_tags':False}}}}
        row=['Ahirwadi','','-a','', '', '', '', 'fixture[p. 1]', '', '', 'fixture:1', '', '', '', 'suffix']
        stream=io.StringIO();csv.writer(stream).writerow(row)
        payload={'csv':stream.getvalue().encode(),'yaml':yaml.safe_dump(settings).encode(),
            'audit':(json.dumps({'source_unit_key':'fixture:1','entry_keys':['fixture:1'],'accepted_for_installation':True,'source_record':{'raw_form':'-a'}})+'\n').encode(),
            'profile':b'Grapheme\tIPA\n-\t\na\ta\n','bib':(prior_bib+refs).encode(),
            'dialects':(prior_dialects+own_dialects.split('\n',1)[1]).encode()}
        files={'installation/install_source.py':m.file_hash(local/'install_source.py')}
        assets={}; baselines={}
        for role,target in m.TARGETS.items():
            stage=local/(m.STEM+'.yaml' if role=='yaml' else role+'.asset');stage.write_bytes(payload[role])
            rel=str(stage.relative_to(package));files[rel]=m.file_hash(stage)
            assets[role]={'target':target,'input':rel,'output_sha256':files[rel]}
            targetpath=data/target;targetpath.parent.mkdir(parents=True,exist_ok=True)
            if role=='bib':targetpath.write_text(prior_bib)
            elif role=='dialects':targetpath.write_text(prior_dialects)
            baselines[target]=m.file_hash(targetpath)
        for name in ['baseline-bibliography.bib','baseline-dialects.csv','staged-references.bib','staged-dialects.csv']:
            files['installation/'+name]=m.file_hash(local/name)
        identity=data/'data/form-identities.csv';identity.parent.mkdir(parents=True,exist_ok=True);identity.write_text('fixture durable identity\n')
        original=workspace/'tmp/original';original.parent.mkdir();original.write_text('fixture original\n')
        a={'installation_approved':True,'result':'pass','shared_baseline_captured':True,'audit_acceptance_ready':True,
            'assets':assets,'files':files,'canonical_baselines':baselines,'original_inputs':{'tmp/original':m.file_hash(original)},
            'durable_identity_sha256':m.file_hash(identity),'rows':1,'audit_units':1,
            'variant_edges':0,'borrowed_edges':0,'component_children':0,
            'own_reference_keys':['fixture'],'approved_reference_replacements':[],'approved_dialect_replacements':[]}
        m.APPROVAL.write_text(json.dumps(a))
        yield m,a,payload

def test_actual_payload_dry_run_install_repeat_after_shared_advancement(transaction):
    m,a,payload=transaction
    assert m.main()['canonical_mutations'] is False
    assert not (m.HERE/'installation-transaction.json').exists()
    assert m.main(install=True)['canonical_mutations'] is True
    assert m.file_hash(m.DATA/'data/form-identities.csv')==a['durable_identity_sha256']
    with (m.DATA/m.TARGETS['bib']).open('a') as f:f.write('@misc{later,title={Later source}}\n')
    with (m.DATA/m.TARGETS['dialects']).open('a') as f:f.write('later,Ahirwadi,Later\n')
    (m.DATA/'data/form-identities.csv').write_text('legitimate later identity build\n')
    before={name:m.file_hash(m.DATA/name) for name in m.TARGETS.values()}
    assert m.main(install=True)['canonical_mutations'] is False
    assert before=={name:m.file_hash(m.DATA/name) for name in m.TARGETS.values()}

def test_stale_input_pin_fails_before_any_mutation(transaction):
    m,a,payload=transaction;(m.HERE/'csv.asset').write_bytes(b'drift')
    with pytest.raises(AssertionError,match='approved input drift'):m.main(install=True)
    assert not (m.HERE/'installation-transaction.json').exists()

def test_false_root_approval_fails_closed(transaction):
    m,a,payload=transaction;a['installation_approved']=False;m.APPROVAL.write_text(json.dumps(a))
    with pytest.raises(AssertionError,match='root acceptance required'):m.main(install=True)

def test_original_drift_fails_closed(transaction):
    m,a,payload=transaction;(m.DATA.parent/'tmp/original').write_text('changed')
    with pytest.raises(AssertionError,match='original drift'):m.main(install=True)

def test_canonical_shared_drift_fails_without_overwrite(transaction):
    m,a,payload=transaction;(m.DATA/m.TARGETS['bib']).write_text('external change')
    with pytest.raises(AssertionError,match='canonical baseline drift'):m.main(install=True)
    assert not (m.HERE/'installation-transaction.json').exists()

@pytest.mark.parametrize('role',['bib','dialects'])
def test_unrelated_records_cannot_be_changed_in_planned_output(transaction,role):
    m,a,payload=transaction
    revised=dict(payload);revised[role]=revised[role].replace(b'Old',b'Altered')
    with pytest.raises(AssertionError,match='unrelated'):m.validate_assets(a,revised)

def test_own_shared_member_drift_rejects_repeat(transaction):
    m,a,payload=transaction;m.main(install=True)
    p=m.DATA/m.TARGETS['bib'];p.write_text(p.read_text().replace('Approved source','Changed source'))
    with pytest.raises(AssertionError,match='own bibliography drift'):m.main(install=True)

def test_atomic_replace_failure_preserves_target_and_resumes(transaction,monkeypatch):
    m,a,payload=transaction;replace=Path.replace;state={'failed':False}
    def fail(path,target):
        if Path(target)==m.DATA/m.TARGETS['bib'] and not state['failed']:
            state['failed']=True;raise OSError('fixture interruption before atomic replacement')
        return replace(path,target)
    monkeypatch.setattr(Path,'replace',fail)
    with pytest.raises(OSError):m.main(install=True)
    assert m.file_hash(m.DATA/m.TARGETS['bib'])==a['canonical_baselines'][m.TARGETS['bib']]
    assert not (m.HERE/'installation-receipt.json').exists()
    assert m.main(install=True)['canonical_mutations'] is True

def test_partial_journal_recovery_after_completed_shared_write(transaction,monkeypatch):
    m,a,payload=transaction;replace=Path.replace;state={'failed':False}
    def fail(path,target):
        result=replace(path,target)
        if Path(target)==m.DATA/m.TARGETS['bib'] and not state['failed']:
            state['failed']=True;raise OSError('fixture interruption after replacement')
        return result
    monkeypatch.setattr(Path,'replace',fail)
    with pytest.raises(OSError):m.main(install=True)
    assert not (m.HERE/'installation-receipt.json').exists()
    assert m.main(install=True)['canonical_mutations'] is True

def test_postwrite_external_drift_never_receipted_or_overwritten(transaction,monkeypatch):
    m,a,payload=transaction;replace=Path.replace
    def drift(path,target):
        result=replace(path,target)
        if Path(target)==m.DATA/m.TARGETS['dialects']:(m.DATA/m.TARGETS['csv']).write_bytes(b'external edit')
        return result
    monkeypatch.setattr(Path,'replace',drift)
    with pytest.raises(AssertionError,match='post-write drift'):m.main(install=True)
    assert not (m.HERE/'installation-receipt.json').exists()
    with pytest.raises(AssertionError,match='partial transaction drift'):m.main(install=True,rollback=True)

def test_rollback_exactly_restores_all_six_baselines(transaction):
    m,a,payload=transaction;m.main(install=True);m.main(install=True,rollback=True)
    assert all(m.file_hash(m.DATA/name)==pin for name,pin in a['canonical_baselines'].items())

def test_real_scaffold_has_no_authority():
    spec=importlib.util.spec_from_file_location('ahirwati_disabled',HERE/'install_source.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert json.loads((HERE/'root-acceptance-draft.json').read_text())['installation_approved'] is False
    assert not m.APPROVAL.exists()
    with pytest.raises(FileNotFoundError):m.validate_inputs()
