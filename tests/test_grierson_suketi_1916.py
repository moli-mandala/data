"""Canonical whole-source parity and metadata, without a database build."""
import csv, hashlib, importlib.util, json
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/grierson_suketi_1916'
CSV=DATA/'data/other/forms/20260925-grierson-suketi.csv'
def test_canonical_reviewed_parity_and_regeneration():
 spec=importlib.util.spec_from_file_location('suketi_canonical',P/'import_source.py')
 source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
 rows,audit=source.generate()
 assert rows==list(csv.reader(CSV.open())) and len(rows)==353
 assert audit==[json.loads(x) for x in (P/'audit.jsonl').read_text().splitlines()] and len(audit)==440
 report=json.loads((P/'independent-full-audit-20260926-pass1.json').read_text())
 assert report['state']=='passed_independent_source_sample' and report['sample_size']==20
 for name,h in report['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
 for proposal,installed in [('proposal.csv',CSV),('proposal-audit.jsonl',P/'audit.jsonl'),('proposal-profile.txt',DATA/'conversion/grierson-suketi-1916.txt')]:assert (P/proposal).read_bytes()==installed.read_bytes()
 legacy=list(csv.reader((P/'legacy-pilot/20260925-grierson-suketi.csv').open()))
 assert len(legacy)==55 and {r[10] for r in legacy}<={r[10] for r in rows}
def test_canonical_metadata_profile_policy_and_source_graph():
 import profile_policy,source_meta
 assert source_meta.SourceMeta().transcription('grierson1916suketi',CSV,'suk')[0]=='grierson-suketi-1916'
 assert 'grierson-suketi-1916' not in profile_policy.audit({})
 rows=list(csv.reader(CSV.open()))
 assert all(r[0]=='suk' and r[7].startswith('grierson1916suketi[') for r in rows)
 assert all(not any(r[11:14]) for r in rows)
 assert '@book{grierson1916suketi,' in (DATA/'cldf/sources.bib').read_text()
 assert 'suk' in {r[0] for r in csv.reader((DATA/'cldf/languages.csv').open())}
