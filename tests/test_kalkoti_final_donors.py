import csv, json, importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/other/params/raw_data'
def test_final_donors_reproduce_and_preserve_source():
 spec=importlib.util.spec_from_file_location('final_donors',RAW/'kalkoti_final_donors.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.OUTPUT.read_bytes()==m.render().encode()
 a=json.loads((RAW/'20260909-kalkoti-final-donors-audit.json').read_text())
 assert [(r['Language_ID'],r['Form']) for r in a]==[('H','hāsil'),('Psht','kedāy ši')]
 assert a[1]['Original']=='kedáy si' and a[1]['Native'].endswith('شي')
 assert all(r['Source'] and '\ufffd' not in r['Form'] for r in a)
def test_two_final_borrowings_compile():
 p=json.loads((RAW/'20260909-kalkoti-approved-batch18.json').read_text())
 f={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open())}
 e={(r['Child_ID'],r['Parent_ID'],r['Kind'],r['Rank']) for r in csv.DictReader((ROOT/'cldf/edges.csv').open())}
 assert len(p)==2
 for r in p:
  assert f[r['parent']]['Status']=='entry'
  assert f[r['parent']]['Language_ID']==r['donor']['language']
  for fid in r['formIds']:
   assert f[fid]['Status']==''
   assert (fid,r['parent'],'borrowed','1') in e
 refs={r['ID'] for r in csv.DictReader((ROOT/'cldf/references.csv').open())}
 for r in p:
  assert all(s.split('[',1)[0] in refs for s in r['evidenceSource'].split(';'))
