"""Canonical whole-source Bhadrawahi installation checks."""
import csv,importlib.util,io,json,unicodedata
from pathlib import Path
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_bhadrawahi_1908'
CSV=DATA/'data/other/forms/20260925-bailey-bhadrawahi.csv'
s=importlib.util.spec_from_file_location('bhadrawahi_install',P/'import_source.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_installed_regeneration_and_frozen_audit():
 rows,audit=m.generate()
 assert rows==list(csv.reader(CSV.open()))
 assert audit==[json.loads(x) for x in (P/'audit.jsonl').read_text().splitlines()]
 assert CSV.read_bytes()==(P/'full-staged.csv').read_bytes()
 assert len(rows)==709 and len(audit)==623
 assert {r[10] for r in csv.reader((P/'legacy-pilot.csv').open())}<={r[10] for r in rows}
 assert (P/'independent-full-audit-20260926-pass1.json').exists()
def test_installed_metadata_profile_registry_and_parse():
 import source_meta,profile_policy,make_cldf,tags
 rows=list(csv.reader(CSV.open()));t=Tokenizer(str(DATA/'conversion/bailey-bhadrawahi-1908.txt'))
 assert source_meta.SourceMeta().transcription('bailey1908bhadrawahi',CSV,'bhad')[0]=='bailey-bhadrawahi-1908'
 assert 'bailey-bhadrawahi-1908' not in profile_policy.audit({})
 assert 'bhad' in {r[0] for r in csv.reader((DATA/'cldf/languages.csv').open())}
 assert '@book{bailey1908bhadrawahi,' in (DATA/'cldf/sources.bib').read_text()
 for r in rows:
  assert len(r)==15 and r[0]=='bhad'
  assert all(r[i]=='' for i in (1,4,5,8,9,11,12,13))
  assert set(r[14].split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
  assert unicodedata.normalize('NFC',t(r[2],column='IPA').replace(' ','').replace('#',' '))==r[2].replace('w','v').replace('ṅ','ŋ')
 e=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),e,name='20260925-bailey-bhadrawahi')
 assert not e.getvalue() and len(parsed)==stats['converted']==709
 raw={r[10]:r for r in rows}
 assert {r.entry_key for r in parsed}==set(raw)
 assert all(r.old_form==raw[r.entry_key][2] for r in parsed)
