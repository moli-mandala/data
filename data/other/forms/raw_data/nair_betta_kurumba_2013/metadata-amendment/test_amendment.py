"""Small isolated single-target metadata and drift guards."""
import importlib.util
import json
from pathlib import Path
import pytest
P=Path(__file__).resolve().parent

def load():
    spec=importlib.util.spec_from_file_location('betta_amendment',P/'import_source.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_single_target_amendment_and_exact_noop(tmp_path):
    m=load();p=tmp_path/'yaml';p.write_bytes(b'baseline')
    assert m.replace_yaml(p,b'approved',m.digest(b'baseline'))=='metadata amendment installed'
    assert p.read_bytes()==b'approved'
    assert m.replace_yaml(p,b'approved',m.digest(b'baseline'))=='already installed; exact no-op'
    assert list(tmp_path.iterdir())==[p]

def test_damaged_yaml_refused_without_mutation(tmp_path):
    m=load();p=tmp_path/'yaml';p.write_bytes(b'user change')
    with pytest.raises(AssertionError,match='unexpected YAML drift'):
        m.replace_yaml(p,b'approved',m.digest(b'baseline'))
    assert p.read_bytes()==b'user change'

def test_own_source_assets_and_bib_membership(tmp_path,monkeypatch):
    m=load();targets={k:tmp_path/k for k in ('csv','audit','bib')}
    for k in ('csv','audit'): targets[k].write_text(k)
    targets['bib'].write_text((P.parent/'proposed-reference.bib').read_text()+'\n@misc{later_source,title={Later source}}\n')
    receipt=tmp_path/'receipt';receipt.write_text(json.dumps({'outputs':{k:m.legacy.current(v) for k,v in targets.items()}}))
    monkeypatch.setattr(m.legacy,'TARGETS',targets);monkeypatch.setattr(m.legacy,'RECEIPT',receipt)
    monkeypatch.setattr(m.legacy,'validate_approval',lambda:None)
    m.validate_own_assets()
    targets['audit'].write_text('damaged')
    with pytest.raises(AssertionError,match='source asset drift'):m.validate_own_assets()
    targets['audit'].write_text('audit')
    targets['bib'].write_text(targets['bib'].read_text().replace('Jayapal','Different'))
    with pytest.raises(AssertionError,match='source bibliography drift'):m.validate_own_assets()
