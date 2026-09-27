"""Focused checks for the complete independently reviewed Bailey Padari source stage."""
import csv,hashlib,importlib.util,io,json,sys,unicodedata
from collections import Counter
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
PACKAGE=DATA/'data/other/forms/raw_data/bailey_padari_1908'
CSV=DATA/'data/other/forms/20260925-bailey-padari.csv'
PROFILE=DATA/'conversion/bailey-padari-1908.txt'
spec=importlib.util.spec_from_file_location('bailey_padari_1908',PACKAGE/'import_source.py')
source=importlib.util.module_from_spec(spec);sys.modules[spec.name]=source;spec.loader.exec_module(source)
def installed():return list(csv.reader(CSV.open()))
def audited():return [json.loads(x) for x in (PACKAGE/'audit.jsonl').read_text().splitlines()]
def test_complete_scope_and_regeneration():
 rows,audit=source.generate()
 assert rows==installed() and audit==audited()
 assert len(rows)==768 and len(audit)==753
 assert Counter(a['status'] for a in audit)=={'ingested':748,'source-blank':5}
 assert len({r[10] for r in rows})==768
 legacy=list(csv.reader((PACKAGE/'legacy-pilot/20260925-bailey-padari.csv').open()))
 assert len(legacy)==13 and {r[10] for r in legacy}<={r[10] for r in rows}
def test_source_mapping_and_local_edges():
 rows=installed();keys={r[10] for r in rows}
 assert all(r[0]=='Padri' for r in rows)
 assert sum(bool(r[11]) for r in rows)==20
 assert all(not r[11] or r[11] in keys for r in rows)
 assert {'fox','hair'}<={r[3] for r in rows}
 assert sum(bool(a['uncertainty']) for a in audited())==2
 assert all(a['review']=='visual-second-pass' for a in audited())
def test_independent_original_review_frozen_hashes():
 report=json.loads((PACKAGE/'independent-full-audit-20260926-pass3.json').read_text())
 assert report['status']=='passed_independent_source_sample' and len(report['sample'])==20
 for name,h in report['hashes'].items():assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==h
 assert CSV.read_bytes()==(PACKAGE/'proposal.csv').read_bytes()
 assert PROFILE.read_bytes()==(PACKAGE/'proposal-profile.txt').read_bytes()
def test_profile_metadata_and_parse():
 import make_cldf,profile_policy,source_meta
 tokenizer=Tokenizer(str(PROFILE))
 for row in installed():
  assert len(row)==15
  original=unicodedata.normalize('NFC',row[2])
  assert tokenizer(original,column='IPA').replace(' ','').replace('#',' ')==original
 assert source_meta.SourceMeta().transcription('bailey1908padari',CSV,'Padri')[0]=='bailey-padari-1908'
 assert 'bailey-padari-1908' not in profile_policy.audit({})
 assert '@book{bailey1908padari,' in (DATA/'cldf/sources.bib').read_text()
 assert 'Padri' in {r[0] for r in csv.reader((DATA/'cldf/languages.csv').open())}
 errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name='20260925-bailey-padari')
 assert not errors.getvalue()
 assert len(parsed)==stats['converted']==768
 raw={r[10]:r for r in installed()}
 assert {r.entry_key for r in parsed}==set(raw)
 assert all(r.old_form==raw[r.entry_key][2] for r in parsed)
