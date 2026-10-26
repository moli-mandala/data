"""Isolated approval and transaction tests; never install into real canonical data."""
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

BASE=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/nair_muduga_2013'

@pytest.fixture
def engine(tmp_path):
    spec=importlib.util.spec_from_file_location('muduga_transaction',BASE/'import_source.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    data=tmp_path/'data';here=data/'data/other/forms/raw_data/nair_muduga_2013';here.mkdir(parents=True)
    for n in (*m.INPUTS.values(),'import_source.py','scope/legacy-baseline.jsonl'):
        p=here/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/n,p)
    # Only isolated fixtures may simulate lexical approval. Real draft remains disabled.
    audit_path=here/m.INPUTS['audit']
    audit=[json.loads(x) for x in audit_path.read_text().splitlines()]
    for unit in audit:unit['accepted_for_source_stage']=True
    audit_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in audit))
    m.DATA=data;m.HERE=here;m.APPROVAL=here/'source-stage-authorization.json';m.RECEIPT=here/'installation-receipt.json';m.JOURNAL=here/'installation-transaction.json'
    m.IDENTITY=data/'data/form-identities.csv';m.IDENTITY.parent.mkdir(parents=True,exist_ok=True);m.IDENTITY.write_text('immutable identity fixture\n')
    baseline=[json.loads(x) for x in (here/'scope/legacy-baseline.jsonl').read_text().splitlines()]
    for name in {x['file'] for x in baseline}:
        with (data/'data/other/forms'/name).open('w',newline='') as f:
            csv.writer(f).writerows(x['fields'] for x in baseline if x['file']==name)
    m.TARGETS={'csv':data/f'data/other/forms/{m.STEM}.csv','yaml':data/f'data/other/forms/{m.STEM}.yaml','audit':here/'installed-audit.jsonl','dialects':data/'cldf/dialects.csv','bib':data/'cldf/sources.bib'}
    (data/'cldf').mkdir();(data/'cldf/languages.csv').write_text('ID,Glottocode\nMuduga,mudu1239\nMalayalam,mala1464\n')
    with m.TARGETS['dialects'].open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(next(csv.reader((here/m.INPUTS['dialects']).open())))
        writer.writerow(['existing_muduga','dialect:Muduga:existing:Existing','Muduga','existing','Existing','','','','S. Dravidian I','fixture',''])
    m.TARGETS['bib'].write_text('@misc{existing, title = {{Existing Authority}}}\n')
    source=data.parent/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf';source.parent.mkdir(parents=True);source.write_bytes(b'isolated source fixture')
    approval={'installation_approved':True,'result':'pass','rows':204,'audit_units':128,'witness_slots':229,'files':{n:m.file_hash(here/n) for n in (*m.INPUTS.values(),'import_source.py','scope/legacy-baseline.jsonl')},'source_pdf_sha256':m.file_hash(source),'durable_identity_sha256':m.file_hash(m.IDENTITY),'canonical_baselines':{str(p.relative_to(data)):m.file_hash(p) for p in m.TARGETS.values()},'deferred':['full compiled/browser gates']}
    m.APPROVAL.write_text(json.dumps(approval));return m

def test_disabled_approval_writes_nothing(engine):
    a=json.loads(engine.APPROVAL.read_text());a['installation_approved']=False;engine.APPROVAL.write_text(json.dumps(a))
    with pytest.raises(AssertionError):engine.main(True)
    assert not engine.JOURNAL.exists() and not engine.RECEIPT.exists()
    assert not engine.TARGETS['csv'].exists()

def test_input_drift_rejected_before_writes(engine):
    (engine.HERE/engine.INPUTS['csv']).write_text('drift')
    with pytest.raises(AssertionError):engine.main(True)
    assert not engine.JOURNAL.exists()

def test_five_targets_and_same_source_repeat_noop(engine):
    plan=engine.main(False);assert len(plan['installed_sha256'])==5 and not engine.JOURNAL.exists()
    engine.main(True)
    before={str(p):p.read_bytes() for p in engine.TARGETS.values()}
    result=engine.main(True)
    assert result['canonical_mutations'] is False
    assert before=={str(p):p.read_bytes() for p in engine.TARGETS.values()}
    assert engine.file_hash(engine.IDENTITY)==plan['durable_identity_sha256']

def test_repeat_preserves_later_shared_contributions(engine):
    engine.main(True)
    with engine.TARGETS['bib'].open('a') as f:f.write('@misc{later, title = {{Later Authority}}}\n')
    before=engine.TARGETS['bib'].read_bytes();engine.main(True)
    assert engine.TARGETS['bib'].read_bytes()==before

def test_interrupted_transaction_recovers_exactly(engine,monkeypatch):
    original=Path.replace
    def fail_audit(path,target):
        if path.name=='installed-audit.jsonl.muduga-install-tmp':raise OSError('fixture interruption')
        return original(path,target)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'replace',fail_audit)
        with pytest.raises(OSError):engine.main(True)
    assert engine.JOURNAL.exists() and not engine.RECEIPT.exists()
    assert engine.TARGETS['csv'].exists() and engine.TARGETS['yaml'].exists()
    assert not engine.TARGETS['audit'].exists()
    result=engine.main(True)
    assert engine.RECEIPT.exists()
    assert all(engine.file_hash(engine.DATA/name)==h for name,h in result['installed_sha256'].items())

def test_installed_asset_drift_is_not_overwritten(engine):
    engine.main(True);engine.TARGETS['csv'].write_text('unapproved replacement')
    with pytest.raises(AssertionError):engine.main(True)
    assert engine.TARGETS['csv'].read_text()=='unapproved replacement'

def test_rejected_source_flags_cannot_be_reapproved_by_hash_only(engine):
    p=engine.HERE/engine.INPUTS['yaml'];p.write_text(p.read_text().replace('ocr: false','ocr: true'))
    a=json.loads(engine.APPROVAL.read_text());a['files'][engine.INPUTS['yaml']]=engine.file_hash(p);engine.APPROVAL.write_text(json.dumps(a))
    with pytest.raises(AssertionError):engine.main(False)
    assert not engine.JOURNAL.exists()

def test_legacy_source_drift_rejected_before_writes(engine):
    p=engine.DATA/'data/other/forms/20260912-muduga.csv'
    p.write_text('unreviewed old-source replacement')
    with pytest.raises(AssertionError,match='existing source drift'):engine.main(True)
    assert not engine.JOURNAL.exists()

def test_pending_lexical_acceptance_cannot_install(engine):
    p=engine.HERE/engine.INPUTS['audit']
    units=[json.loads(x) for x in p.read_text().splitlines()]
    units[0]['accepted_for_source_stage']=False
    p.write_text(''.join(json.dumps(x)+'\n' for x in units))
    a=json.loads(engine.APPROVAL.read_text());a['files'][engine.INPUTS['audit']]=engine.file_hash(p)
    engine.APPROVAL.write_text(json.dumps(a))
    with pytest.raises(AssertionError):engine.main(True)
    assert not engine.JOURNAL.exists()

def test_disabled_plan_validates_without_authorizing_real_transaction(engine):
    a=json.loads(engine.APPROVAL.read_text());a['installation_approved']=False
    engine.APPROVAL.write_text(json.dumps(a))
    _,planned=engine.prepare(require_approval=False)
    assert len(planned)==5 and not engine.JOURNAL.exists() and not engine.RECEIPT.exists()
    with pytest.raises(AssertionError):engine.main(True)
