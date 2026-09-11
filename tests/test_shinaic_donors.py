import csv,importlib.util,json,os,unicodedata
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'data/other/params/raw_data'
def audit():return json.loads((RAW/'20260910-shinaic-donors-audit.json').read_text())
def test_selection():
 s=importlib.util.spec_from_file_location('shinaic_donors',RAW/'shinaic_donors.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 assert m.OUTPUT.read_bytes()==m.render().encode()
 a=audit();assert len(a)==406;assert sum(len(x['uses']) for x in a)==416
 assert len({x['ID'] for x in a})==406
 langs={r['ID'] for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
 for x in a:
  assert x['Language_ID'] in langs and x['Evidence'] and x['Notes']
  assert x['Form']==unicodedata.normalize('NFC',x['Form']) and '\ufffd' not in x['Form']
  assert '[' in x['Source'] and x['Source'].count('[')==x['Source'].count(']')
def test_homonyms():
 d={(u['language'],u['proposal']):a for a in audit() for u in a['uses']}
 assert '34798' in d['Ush',200]['Source']
 assert 'potato' in d['Ush',206]['Gloss']
 assert '14945' in d['Sv',350]['Source']
 assert d['Phal',81]['Original']=='aayanda, aainda'
 assert 'source-attributed' in d['Phal',81]['Scope']
def test_compiled():
 b=os.environ.get('SHINAIC_BUILD')
 if not b:pytest.skip('Requires isolated compiled build')
 f={r['ID']:r for r in csv.DictReader(open(Path(b)/'cldf/forms.csv'))}
 aliases={r['Legacy_ID']:r['Form_ID'] for r in csv.DictReader(open(Path(b)/'cldf/form-id-aliases.csv'))}
 refs={r['ID'] for r in csv.DictReader(open(Path(b)/'cldf/references.csv'))}
 for a in audit():
  r=f[aliases[a['ID']]]
  assert r['Language_ID']==a['Language_ID'] and r['Form']==a['Form'] and r['Gloss']==a['Gloss'] and r['Status']=='entry'
  assert a.get('Persistent_ID',r['ID'])==r['ID']
  assert all(c.split('[',1)[0] in refs for c in r['Source'].split(';'))


def test_saved_graph():
 b=os.environ.get('SHINAIC_BUILD')
 if not b:pytest.skip('Requires isolated compiled build')
 E={(r['Child_ID'],r['Parent_ID'],r['Kind'],r['Rank'],r['Pos']) for r in csv.DictReader(open(Path(b)/'cldf/edges.csv'))}
 A=list(csv.DictReader((ROOT/'data/etymology-assignments.csv').open()))
 for l,batch in [('Sv','010'),('Ush','001'),('Phal','001')]:
  m=json.loads((ROOT/f'curation/etymology-lab/{l}/batch-{batch}.json').read_text())
  assert m['status']=='saved'
  for p in m['proposals']:
   assert p['saveStatus']=='saved' and p['assignments']
   for a in p['assignments']:
    assert a in A
    assert (a['Form_ID'],a['Etymon_ID'],a['Kind'],a['Rank'],a['Pos']) in E
