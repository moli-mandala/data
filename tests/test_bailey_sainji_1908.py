"""Installed whole-chapter Sainji checks; no database generation."""
import csv,importlib.util,io,json,sys,unicodedata
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_sainji_1908'
CSV=DATA/'data/other/forms/20260925-bailey-sainji.csv'
PROFILE=DATA/'conversion/bailey-sainji-1908.txt'
s=importlib.util.spec_from_file_location('sainji_installed',P/'import_source.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_installed_authority_reconciliation():
 rows,audit=m.generate()
 assert rows==list(csv.reader(CSV.open()))==list(csv.reader((P/'full-staged.csv').open()))
 assert audit==[json.loads(x) for x in (P/'audit.jsonl').read_text().splitlines()]
 assert len(rows)==270 and len(audit)==241
 assert len({r[10] for r in rows})==270
 old=list(csv.reader((P/'legacy-pilot.csv').open()))
 assert len(old)==55 and {r[10] for r in old}<={r[10] for r in rows}
 assert all(r[0]=='sai' and r[11]==r[12]==r[13]=='' for r in rows)
def test_source_metadata_profile_and_parse():
 import source_meta,profile_policy,make_cldf
 assert source_meta.SourceMeta().transcription('bailey1908sainji',CSV,'sai')[0]=='bailey-sainji-1908'
 assert 'bailey-sainji-1908' not in profile_policy.audit({})
 assert '@book{bailey1908sainji,' in (DATA/'cldf/sources.bib').read_text()
 assert 'sai' in {r[0] for r in csv.reader((DATA/'cldf/languages.csv').open())}
 tokenizer=Tokenizer(str(PROFILE));raw=list(csv.reader(CSV.open()))
 for r in raw:
  actual=unicodedata.normalize('NFC',tokenizer(r[2],column='IPA').replace(' ','').replace('#',' '))
  assert actual==r[2].replace('w','v').replace('ṅ','ŋ')
 e=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),e,name='20260925-bailey-sainji')
 assert not e.getvalue() and len(parsed)==stats['converted']==270
 assert {r.entry_key for r in parsed}=={r[10] for r in raw}
 assert all(not r[11] for r in raw)
