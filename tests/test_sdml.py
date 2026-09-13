import csv, importlib.util, json, unicodedata
from pathlib import Path
from segments.tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sdml',ROOT/'data/other/forms/raw_data/sdml.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def rows():return list(csv.reader((ROOT/'data/other/forms/20260911-sdml.csv').open()))
def test_snapshot_counts_and_reproduction(tmp_path):
 r=m.propose(tmp_path)
 assert r['statuses']=={'unlinked':47317,'ambiguous':33,'skipped':606}
 assert r['raw_cells']==19783 and r['tokens']==47956
 assert (tmp_path/'20260911-sdml.csv').read_bytes()==(ROOT/'data/other/forms/20260911-sdml.csv').read_bytes()
def test_keys_and_dialects():
 rs=rows();assert len({r[10] for r in rs})==47317
 registry={r['Tag']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
 assert len({r[14].split()[0] for r in rs})==269
 for r in rs:
  assert len(r)==15 and r[0]=='M' and not r[1] and not r[4] and r[5]==r[2]
  d=registry[r[14].split()[0]];assert d['Language_ID']=='M' and d['Quality']=='A'
  assert 15<float(d['Latitude'])<23 and 72<float(d['Longitude'])<82
  assert r[7].startswith('sdml2026[') and r[10].startswith('sdml2026:')
def test_profile_coverage():
 t=Tokenizer(str(ROOT/'conversion/sdml.txt'))
 for r in rows():assert '�' not in t(r[2],column='IPA')
 assert t('c č j ǰ',column='IPA').replace(' ','').replace('#',' ')=='c č j ǰ'
def test_gloss_and_exclusions():
 assert m.gloss('female_egos_brothers_daughter')=="female ego's brother's daughter"
 audit=[json.loads(l) for l in (m.RAW/'20260911-sdml-audit.jsonl').open()]
 assert any(a['raw_token']=='NA' and a['status']=='skipped' for a in audit)
 assert any(a['raw_token']=='attya (3) atti (6) mawḷəṇ' and a['status']=='ambiguous' for a in audit)
 assert all(a['frequency'] is None for a in audit if 'frequency:missing-or-unmatched-list' in a['reasons'])
def test_compiled_source():
 forms=[r for r in csv.DictReader((ROOT/'cldf/forms.csv').open()) if r['Source'].startswith('sdml2026[')]
 assert len(forms)==47317
 assert all(r['Status']=='unlinked' and r['Language_ID']=='M' for r in forms)
 assert all(r['Original'] and r['Phonemic'] for r in forms)
 ids={r['ID'] for r in forms}
 assert not any(r['Child_ID'] in ids for r in csv.DictReader((ROOT/'cldf/edges.csv').open()))
