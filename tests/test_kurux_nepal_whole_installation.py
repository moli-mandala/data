"""Installed routes and immutable baseline; no compiled database needed."""
import csv,hashlib,io,json,unicodedata
from pathlib import Path
import make_cldf,make_refs,source_meta
from pybtex.database import parse_file
import pybtex
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data/other/forms/raw_data/kurux_nepal_2022/whole-source-recovery'
ROUTES={'wordlist':'20260813-kurux-nepal','grammar':'20260927-kurux-nepal-grammar','native':'20260927-kurux-nepal-native','roman':'20260927-kurux-nepal-roman'}
SOURCE='shackelford-swenson-chaudhary-maggard2022kurux'
def test_installed_routes_parse_exact_reviewed_rows():
 keys=set();count=0
 for bucket,stem in ROUTES.items():
  path=ROOT/f'data/other/forms/{stem}.csv'
  assert path.read_bytes()==(P/f'proposal/{bucket}.csv').read_bytes()
  raw={r[10]:r for r in csv.reader(path.open())}
  err=io.StringIO();parsed,_=make_cldf.parse_file(str(path),errors=err)
  assert not err.getvalue(),err.getvalue()
  assert len(parsed)==len(raw)
  for r in parsed:
   x=raw[r.entry_key]
   assert r.old_form==x[2] and r.native==x[4] and r.ipa==x[5]
   assert set(x[14].split())<=set(r.tags.split())
   assert '\ufffd' not in r.form
   if bucket in ('native','roman'):assert unicodedata.normalize('NFC',r.form)==unicodedata.normalize('NFC',x[2])
   assert set(make_refs.source_ids(x[7]))=={SOURCE}
  keys.update(raw);count+=len(raw)
 assert count==len(keys)==2465
 legacy={r[10] for r in csv.reader((P/'legacy-before-whole-recovery.csv').open())}
 assert len(legacy)==1336 and legacy<=keys

def test_registered_profiles_dialects_and_formatted_source():
 for name in ('phonetic','native','roman'):
  p=ROOT/f'conversion/kurux-nepal-whole-{name}.txt'
  assert p.read_bytes()==(P/f'proposal/{p.name}').read_bytes()
 registry={r['ID']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
 for r in csv.DictReader((P/'proposed-dialects.csv').open()):assert registry[r['ID']]==r
 entry=parse_file(str(ROOT/'cldf/sources.bib')).entries[SOURCE]
 assert '338 prompts' in entry.fields['included']
 formatted=pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'),'plain',output_backend='markdown')
 assert '2022' in formatted and 'Shackelford' in formatted
 assert make_refs.create_short_ref(entry)=='S2022'

def test_frozen_outputs_and_identity_receipt():
 m=json.loads((P/'frozen-manifest.json').read_text())
 for name,h in m['files'].items():
  if name.startswith('proposal/'):
   assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
 repair=json.loads((P/'dependency-repair-20260927.json').read_text())
 assert hashlib.sha256((P/'legacy-before-whole-recovery.csv').read_bytes()).hexdigest()==repair['baseline_sha256']
 report=json.loads((P/'installation-report-20260927.json').read_text())
 assert report['legacy_keys_preserved']==1336
 assert report['durable_identity_sha256_unchanged']==hashlib.sha256((ROOT/'data/form-identities.csv').read_bytes()).hexdigest()
