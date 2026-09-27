"""Installed whole-source Rangri parity and metadata, without a full data build."""
import csv,hashlib,importlib.util,json
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/grierson_malvi_rangri_1908'

def load(name):
 spec=importlib.util.spec_from_file_location('rangri_'+name,P/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_complete_installed_importer_parity_and_pilot_keys():
 rows,audit=load('import_source').generate()
 assert rows==list(csv.reader((DATA/'data/other/forms/20260925-grierson-malvi-rangri.csv').open()))
 assert audit==[json.loads(s) for s in (P/'audit.jsonl').read_text().splitlines()]
 legacy=list(csv.reader((P/'legacy-pilot/20260925-grierson-malvi-rangri.csv').open()))
 assert {r[10] for r in legacy}<={r[10] for r in rows}
 assert len(rows)==1170 and len(audit)==1346

def test_frozen_independent_audit_matches_canonical():
 report=json.loads((P/'independent-full-audit-20260926-pass2.json').read_text())
 assert report['status']=='pass' and report['material_errors']==0 and report['sample_size']==20
 for name,h in report['hashes_before_and_after'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h

def test_registered_metadata_profile_and_parsed_fields():
 before=hashlib.sha256((DATA/'data/form-identities.csv').read_bytes()).hexdigest()
 result=load('verify_installed').verify(before)
 assert result['rows']==1170 and result['native_rows']==601
