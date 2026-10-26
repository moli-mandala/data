"""Isolated transaction guards; no canonical source paths are written."""
import importlib.util
import json
import shutil
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent

@pytest.fixture
def engine(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('irula_test_engine',HERE/'install_source.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source=tmp_path/'source';source.mkdir()
    for name in m.INPUTS:shutil.copy2(HERE/name,source/name)
    data=tmp_path/'data';(data/'cldf').mkdir(parents=True)
    (data/'cldf/languages.csv').write_text('ID,Glottocode\nIrula,irul1243\n')
    bib=data/'cldf/sources.bib';bib.write_text('@misc{unrelated,title={Unrelated}}\n')
    ident=data/'identities.csv';ident.write_text('existing durable identities\n')
    targets={'csv':data/'source.csv','yaml':data/'source.yaml','bib':bib,'audit':source/'installed-audit.jsonl'}
    for key,value in [('HERE',source),('DATA',data),('TARGETS',targets),('IDENTITY',ident),('APPROVAL',source/'approval.json'),('RECEIPT',source/'receipt.json')]:monkeypatch.setattr(m,key,value)
    def approve():
        m.APPROVAL.write_text(json.dumps({'status':'approved','material_errors':0,'installer_sha256':m.current(Path(m.__file__)),'source_pdf_sha256':m.PDF_SHA,'input_sha256':{n:m.current(source/n) for n in m.INPUTS}}))
    return m,approve

def test_approval_required_and_dry_run_is_read_only(engine):
    m,_=engine;before=m.TARGETS['bib'].read_bytes()
    m.main(install=False)
    with pytest.raises(AssertionError,match='approval'):m.main(install=True)
    assert m.TARGETS['bib'].read_bytes()==before and not m.RECEIPT.exists()
    assert not m.TARGETS['csv'].exists()

def test_install_repeat_allows_unrelated_bib_and_identity_advance(engine):
    m,approve=engine;approve();m.main(install=True)
    own={k:m.current(v) for k,v in m.TARGETS.items() if k!='bib'}
    with m.TARGETS['bib'].open('a') as f:f.write('@misc{later,title={Later source}}\n')
    m.IDENTITY.write_text('later legitimate identity state\n')
    receipt=m.RECEIPT.read_bytes();m.main(install=True)
    assert own=={k:m.current(v) for k,v in m.TARGETS.items() if k!='bib'}
    assert receipt==m.RECEIPT.read_bytes()
    m.TARGETS['yaml'].write_text('unexpected source edit')
    with pytest.raises(AssertionError):m.main(install=True)

def test_unexpected_target_and_mid_transaction_failure(engine,monkeypatch):
    m,approve=engine;approve()
    m.TARGETS['csv'].write_text('user edit')
    with pytest.raises(AssertionError,match='Unexpected'):m.main(install=True)
    m.TARGETS['csv'].unlink();before=m.TARGETS['bib'].read_bytes()
    replace=m.os.replace
    def fail_bib(src,dst):
        if dst==m.TARGETS['bib']:raise OSError('isolated simulated write failure')
        return replace(src,dst)
    monkeypatch.setattr(m.os,'replace',fail_bib)
    with pytest.raises(OSError):m.main(install=True)
    assert not m.RECEIPT.exists() and not m.TARGETS['csv'].exists() and not m.TARGETS['yaml'].exists()
    assert m.TARGETS['bib'].read_bytes()==before
