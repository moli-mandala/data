"""Canonical whole-source Sikalgari verification, without a database build."""
import csv,hashlib,importlib.util,io,json
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/grierson_sikalgari_1922'
CSV=DATA/'data/other/forms/20260925-grierson-sikalgari.csv'
spec=importlib.util.spec_from_file_location('sikalgari_installed',P/'import_source_full.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)

def test_canonical_matches_audited_whole_stage():
 rows,audit=source.generate()
 assert rows==list(csv.reader(CSV.open())) and len(rows)==633
 assert audit==[json.loads(l) for l in (P/'audit.jsonl').read_text().splitlines()] and len(audit)==790
 report=json.loads((P/'independent-full-audit-20260926-pass1.json').read_text())
 assert report['state']=='passed_independent_source_sample'
 assert hashlib.sha256(CSV.read_bytes()).hexdigest()==report['hashes']['full-staged.csv']
 assert hashlib.sha256((P/'audit.jsonl').read_bytes()).hexdigest()==report['hashes']['full-staged-audit.jsonl']
 old={r[10]:r for r in csv.reader((P/'legacy-installed-before-full.csv').open())};new={r[10]:r for r in rows}
 assert len(old)==82 and old.keys()<=new.keys()
 assert any(old[k][2]!=new[k][2] for k in old)

def test_scoped_parse_profile_metadata_and_graph_contract():
 import make_cldf,profile_policy,source_meta
 assert source_meta.SourceMeta().transcription('grierson1922lsi11',CSV,'Sik')[0]=='grierson-sikalgari-1922'
 assert 'grierson-sikalgari-1922' not in profile_policy.audit({})
 e=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),e,name='20260925-grierson-sikalgari')
 assert not e.getvalue() and len(parsed)==stats['converted']==633
 raw={r[10]:r for r in csv.reader(CSV.open())}
 assert {r.entry_key for r in parsed}==raw.keys()
 assert all(r.old_form==raw[r.entry_key][2] for r in parsed)
 assert all(not r[c] for r in raw.values() for c in (1,4,5,8,9,11,12,13))
 assert (DATA/'cldf/sources.bib').read_text().count('@book{grierson1922lsi11,')==1
