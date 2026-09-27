"""Installed whole-source Simla Siraji checks, without a database build."""
import csv,importlib.util,io,json,hashlib
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/grierson_simla_siraji_1916'
CSV=DATA/'data/other/forms/20260925-grierson-simla-siraji.csv'
spec=importlib.util.spec_from_file_location('simla_installed_full',P/'import_source_full.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)

def test_installed_regeneration_and_legacy_reconciliation():
 rows,audit=source.generate()
 assert rows==list(csv.reader(CSV.open()))
 assert audit==[json.loads(l) for l in (P/'audit.jsonl').read_text().splitlines()]
 assert len(rows)==414 and len(audit)==341
 old={r[10]:r for r in csv.reader((P/'legacy-installed-before-full.csv').open())}
 new={r[10]:r for r in rows}
 assert len(old)==16 and old.keys()<=new.keys()
 assert sum(old[k][2]!=new[k][2] for k in old)==3
 report=json.loads((P/'independent-full-audit-20260926-pass3.json').read_text())
 assert report['literal_errors']==report['metadata_errors']==0
 assert hashlib.sha256(CSV.read_bytes()).hexdigest()==report['hashes']['full-staged.csv']

def test_metadata_profile_scoped_parse_and_reference_graph():
 import make_cldf,profile_policy,source_meta
 assert 'ShimlaSiraji' in {r['ID'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
 assert '@book{grierson1916simlasiraji,' in (DATA/'cldf/sources.bib').read_text()
 assert source_meta.SourceMeta().transcription('grierson1916simlasiraji',CSV,'ShimlaSiraji')[0]=='grierson-simla-siraji-1916'
 assert 'grierson-simla-siraji-1916' not in profile_policy.audit({})
 errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name='20260925-grierson-simla-siraji')
 assert not errors.getvalue()
 assert len(parsed)==stats['converted']==414
 raw={r[10]:r for r in csv.reader(CSV.open())}
 assert {r.entry_key for r in parsed}==raw.keys()
 assert all(r.old_form==raw[r.entry_key][2] for r in parsed)
 assert all(r[0]=='ShimlaSiraji' and r[7].startswith('grierson1916simlasiraji[') for r in raw.values())
 assert all(not r[4] and not r[5] and not r[11] and not r[12] for r in raw.values())
