"""Isolated dry-run tests. Never call main(install=True) or invoke --install."""
import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest
import yaml

SOURCE=Path(__file__).resolve().parents[1]/'import_source.py'


@pytest.fixture
def transaction(tmp_path):
    spec=importlib.util.spec_from_file_location('kui_installer_review',SOURCE)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data=tmp_path/'data'
    here=data/'data/other/forms/raw_data/kui_fixture'
    here.mkdir(parents=True)
    (data/'cldf').mkdir()
    (data/'conversion').mkdir()
    (data/'data/form-identities.csv').write_text('source,key,id\nretained,1,durable\n')
    (data/'cldf/sources.bib').write_text('@book{unrelated, title={Keep {Exact} Text}, author={A, B}, year={1999}}\n')
    header='ID,Tag,Language_ID,Source_Language_ID,Name,Glottocode,Latitude,Longitude,Clade,Location,Quality\n'
    old='old,dialect:Kui:old:Old,Kui,old,Old,,,,S. Dravidian II,,\n'
    (data/'cldf/dialects.csv').write_text(header+old)
    (here/'integration').mkdir()
    (here/'import_source.py').write_bytes(SOURCE.read_bytes())
    row=['Kui','','ā','test','','','source note','grierson1906kui[p. 1]','','','fixture:key','','','','noun']
    with (here/mod.INPUTS['csv']).open('w',newline='') as f:csv.writer(f).writerow(row)
    (here/mod.INPUTS['audit']).write_text(json.dumps(dict(source_unit_key='fixture:key',entry_keys=['fixture:key'],disposition='emitted'))+'\n')
    settings=dict(file=mod.STEM+'.csv',defaults=dict(transcription=dict(profile=mod.PROFILE),identity=dict(dedupe_by_entry_key=True),forms=dict(split_alternates=False)),sources=dict(grierson1906kui=dict(reference=dict(etymology_provenance='source'))))
    (here/mod.INPUTS['yaml']).write_text(yaml.safe_dump(settings))
    (here/mod.INPUTS['profile']).write_text('Grapheme\tIPA\nā\tā\n')
    (here/mod.INPUTS['bib']).write_text('@book{grierson1906kui, title={Reviewed Kui}, author={Grierson, G. A.}, year={1906}}\n')
    (here/mod.INPUTS['dialects']).write_text(header+'new,dialect:Kui:new:New,Kui,new,New,,,,S. Dravidian II,,\n')
    mod.DATA=data
    mod.HERE=here
    mod.APPROVAL=here/'root-installation-approval.json'
    paths=[data/f'data/other/forms/{mod.STEM}.csv',data/f'data/other/forms/{mod.STEM}.yaml',data/f'conversion/{mod.PROFILE}.txt',here/'installed-audit.jsonl',data/'cldf/sources.bib',data/'cldf/dialects.csv']
    approval=dict(installation_approved=True,result='pass',rows=1,audit_units=1,
        files={name:mod.file_hash(here/name) for name in [*mod.INPUTS.values(),'import_source.py']},
        canonical_baselines={str(p.relative_to(data)):mod.file_hash(p) for p in paths},
        durable_identity_sha256=mod.file_hash(data/'data/form-identities.csv'))
    mod.APPROVAL.write_text(json.dumps(approval))
    return mod,approval


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
    with pytest.raises(AssertionError,match='Changed approved input'):mod.prepare()
    source.write_bytes(original)
    (mod.DATA/'data/form-identities.csv').write_text('changed identity')
    with pytest.raises(AssertionError):mod.prepare()


def test_unrelated_bib_and_dialect_header_preserved(transaction):
    mod,_=transaction
    _,planned,_=mod.prepare()
    bib=mod.DATA/'cldf/sources.bib'
    assert planned[bib].startswith(bib.read_bytes().rstrip())
    dialect=mod.DATA/'cldf/dialects.csv'
    values=list(csv.DictReader(io.StringIO(planned[dialect].decode())))
    assert len(values)==2 and [r['ID'] for r in values]==['old','new']
    assert planned[dialect].count(b'ID,Tag,Language_ID')==1


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


@pytest.mark.parametrize('written_count',range(1,7))
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


def test_missing_terminal_newline_is_inserted_before_dialect_append(transaction):
    mod,approval=transaction
    dialect=mod.DATA/'cldf/dialects.csv'
    dialect.write_bytes(dialect.read_bytes().rstrip(b'\r\n'))
    approval['canonical_baselines']['cldf/dialects.csv']=mod.file_hash(dialect)
    mod.APPROVAL.write_text(json.dumps(approval))
    _,planned,_=mod.prepare()
    parsed=list(csv.DictReader(io.StringIO(planned[dialect].decode())))
    assert len(parsed)==2 and all(None not in r for r in parsed)
    assert [r['ID'] for r in parsed]==['old','new']


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


def test_dialect_conflict_is_rejected(transaction):
    mod,approval=transaction
    path=mod.HERE/mod.INPUTS['dialects']
    path.write_text(path.read_text().replace('new,','old,',1))
    approval['files'][mod.INPUTS['dialects']]=mod.file_hash(path)
    mod.APPROVAL.write_text(json.dumps(approval))
    with pytest.raises(AssertionError):mod.prepare()


def test_duplicate_bib_key_is_rejected(transaction):
    mod,approval=transaction
    path=mod.HERE/mod.INPUTS['bib']
    path.write_text(path.read_text()+'@book{unrelated,title={Overwrite}}\n')
    approval['files'][mod.INPUTS['bib']]=mod.file_hash(path)
    mod.APPROVAL.write_text(json.dumps(approval))
    with pytest.raises(AssertionError,match='Proposed reference key already exists'):mod.prepare()
