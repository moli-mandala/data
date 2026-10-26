"""Guarded complete-source reproduction before source-stage installation."""
import importlib.util,json,sys
from pathlib import Path
import pytest,yaml
Q=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/niranjan_chakma_2010/whole-installation'
sys.path.insert(0,str(Q))
spec=importlib.util.spec_from_file_location('chakma_whole_install',Q/'install_whole_source.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_regenerates_frozen_bytes_from_immutable_legacy_baseline():
 result,csvdata,auditdata=m.checked()
 assert result['rows']==3662 and result['preserved_keys']==742
 assert csvdata==(Q.parent/'whole-export-r2/whole-source-export-preview.csv').read_bytes()
 assert auditdata==(Q.parent/'whole-export-r2/whole-source-export-audit.jsonl').read_bytes()
 assert m.CSV!=Q/'legacy-742.csv'

def test_installation_requires_matching_independent_acceptance():
 result,_,_=m.checked()
 if not (Q/'root-installation-approval.json').exists():
  with pytest.raises(RuntimeError,match='No root installation approval'):m.approval(result)
 else:
  m.approval(result)
  with pytest.raises(AssertionError):m.approval({**result,'csv_sha256':'0'*64})

def test_metadata_route_preserves_non_importer_settings():
 current=yaml.safe_load(m.YAML.read_text());proposed=yaml.safe_load((Q/'metadata-proposed.yaml').read_text())
 assert proposed['defaults']['importer']['commands']==[['data/other/forms/raw_data/niranjan_chakma_2010/whole-installation/install_whole_source.py','--install']]
 assert proposed['sources']['niranjan2010chakma']['reference']['etymology_provenance']=='source'
 assert current['sources']['niranjan2010chakma']['reference']['etymology_provenance'] in ('none','source')
 current['sources']['niranjan2010chakma']['reference']['etymology_provenance']='source'
 current['defaults'].pop('importer');proposed['defaults'].pop('importer')
 assert current==proposed
