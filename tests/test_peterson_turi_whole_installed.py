"""Approved whole-source stage: real canonical routes, no database build."""
import csv,hashlib,io,json,runpy,subprocess,sys
from pathlib import Path
import make_cldf,make_refs,source_meta
from pybtex.database import parse_file
from tags import GRAMMATICAL_TAGS
ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'data/other/forms/raw_data/peterson_turi_2024/whole-source-recovery'
OUT=H/'revision4/proposal'
F=ROOT/'data/other/forms'
def rows(p):return list(csv.reader(p.open()))
def test_exact_approved_rows_and_canonical_routes():
 meta=source_meta.load()
 keys=set()
 for p in OUT.glob('*.csv'):
  canonical=F/p.name
  assert canonical.read_bytes()==p.read_bytes()
  raw={r[10]:r for r in rows(canonical)}
  errors=io.StringIO()
  actual,stats=make_cldf.parse_file(str(canonical),errors=errors,name=p.stem)
  assert not errors.getvalue(),errors.getvalue()
  assert {r.entry_key for r in actual}==set(raw)
  for r in actual:
   rr=raw[r.entry_key]
   assert r.old_form==rr[2] and r.ipa==rr[5]
   assert set(rr[14].split())<=set(r.tags.split())
  keys.update(raw)
 assert len(keys)==648
def test_registered_tags_dialects_citations_and_edges():
 dialects={r['Tag']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
 bib=parse_file(str(ROOT/'cldf/sources.bib'))
 allrows=[r for p in OUT.glob('*.csv') for r in rows(p)];keys={r[10] for r in allrows}
 for r in allrows:
  assert set(make_refs.source_ids(r[7]))=={'peterson2024turi'}<=set(bib.entries)
  for t in r[14].split():assert t in dialects or t in GRAMMATICAL_TAGS,t
  assert not r[11] or r[11] in keys
  assert not r[12] and not r[13]
 j=dialects['dialect:Turi:turi_jharkhandi_peterson2024:Jharkhandi']
 assert j['Language_ID']=='Turi' and not j['Latitude'] and not j['Longitude']
 assert '648' in bib.entries['peterson2024turi'].fields['included']
def test_guarded_coordinator_and_historical_entrypoint_are_idempotent():
 watched=[F/p.name for p in OUT.glob('*.csv')]+[F/p.name for p in (H/'installation-overlay').glob('*.yaml')]+[ROOT/'conversion'/p.name for p in OUT.glob('peterson-turi-*.txt')]
 before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
 for script in [H/'install_whole_source.py',H.parent/'import_source.py']:
  subprocess.run([sys.executable,str(script),'--install'],cwd=ROOT,check=True,capture_output=True)
 assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
 routes=[(F/p.name).read_text() for p in (H/'installation-overlay').glob('*.yaml')]
 assert all('whole-source-recovery/install_whole_source.py' in s for s in routes)
 assert sum('  peterson2024turi:' in s for s in routes)==1
