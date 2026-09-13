import csv, importlib.util, json, unicodedata
from pathlib import Path
import pytest
from segments import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ia_dravidian',ROOT/'data/other/forms/raw_data/ia_dravidian.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
COUNTS={'bajjika':1208,'lindgren':3754,'dravlex':2127}
def rows(k):return list(csv.reader((ROOT/f'data/other/forms/20260911-{k}.csv').open()))
@pytest.mark.parametrize('k',COUNTS)
def test_records_and_registry(k):
 rr=rows(k);assert len(rr)==COUNTS[k];assert len({r[10] for r in rr})==len(rr)
 langs={r['ID'] for r in csv.DictReader((ROOT/'cldf/languages.csv').open())};ds={r['Tag']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
 for r in rr:
  assert len(r)==15 and r[0] in langs and r[1]=='' and r[2]
  assert r[7].startswith(mod.SOURCES[k]+'[') and not r[4] and not r[6]
  assert not any(x in r[2] for x in ['�','(cid:','\uf02c','\uf01e'])
  assert unicodedata.normalize('NFC',r[2])==r[2]
  for t in r[14].split():
   if t.startswith('dialect:'):assert ds[t]['Language_ID']==r[0]
@pytest.mark.parametrize('k',COUNTS)
def test_profile_coverage(k):
 name='bajjika' if k=='bajjika' else 'ia-dravidian-ipa';t=Tokenizer(str(ROOT/f'conversion/{name}.txt'))
 for r in rows(k):assert '�' not in t(r[2] if k=='bajjika' else r[5],column='IPA'),r[10]
def test_bajjika_boundaries():
 rr={r[10]:r for r in rows('bajjika')}
 assert rr['bajjika:p103:i14:gaur:v1'][2]=='kehuni'
 assert rr['bajjika:p107:i137:gaur:v2'][2]=='ṭhaṛh'
 assert rr['bajjika:p106:i87:garuda:v1'][2]=='murgi ke bəcca'
 assert rr['bajjika:p109:i173:garuda:v2'][2]=='ekəni səb'
 assert rr['bajjika:p110:i210:gaur:v3'][2]=='u səb'
 assert rr['bajjika:p103:i5:malangawa:v1'][2]=='ãkʰ'
 assert len(mod.extract_bajjika())==210

def test_overlap_all_accounted():
 aa=[json.loads(s) for s in (mod.RAW/'20260911-lindgren-audit.jsonl').read_text().splitlines()]
 assert len(aa)==5881
 reused={a['upstream']['ID'] for a in aa if a['status']=='reused-dravlex'};assert len(reused)==2127
 dd=[json.loads(s) for s in (mod.RAW/'20260911-dravlex-audit.jsonl').read_text().splitlines()]
 assert {i for a in dd for i in a['upstream']['republished_lindgren_ids']}==reused
 assert all('lindgren2023dravidian[' in a['installed'][7] for a in dd)
def test_phonology_examples():
 t=Tokenizer(str(ROOT/'conversion/ia-dravidian-ipa.txt'))
 cv=lambda s:t(s,column='IPA').replace(' ','').replace('#',' ')
 assert cv('maʈːi')=='maṭṭi'
 assert cv('naːɭ')=='nāḷ'
 assert cv('dʒiː')=='ǰī'
 assert cv('a ɖa')=='a ḍa'
 assert cv('maɻai')=='maɻai'

def test_reproducible(tmp_path):
 assert mod.build(tmp_path)['lindgren']['statuses']['reused-dravlex']==2127
 for k in COUNTS:assert (tmp_path/f'20260911-{k}.csv').read_bytes()==(ROOT/f'data/other/forms/20260911-{k}.csv').read_bytes()

def test_compiled_records_preserve_observations_without_graph_edges():
 sources=set(mod.SOURCES.values())
 compiled=[]
 with (ROOT/'cldf/forms.csv').open() as stream:
  for row in csv.DictReader(stream):
   if any(s+'[' in row['Source'] for s in sources):compiled.append(row)
 assert len(compiled)==sum(COUNTS.values())
 for k,n in {'bajjika':1208,'lindgren':5881,'dravlex':2127}.items():
  assert sum(mod.SOURCES[k]+'[' in r['Source'] for r in compiled)==n
 assert all(r['Status']=='unlinked' and not r['Redirect'] for r in compiled)
 assert all(r['Original'] and r['Form'] and '�' not in r['Form'] for r in compiled)
 ids={r['ID'] for r in compiled}
 with (ROOT/'cldf/edges.csv').open() as stream:
  assert not any(r['Child_ID'] in ids or r['Parent_ID'] in ids for r in csv.DictReader(stream))
 with (ROOT/'cldf/form-source-keys.csv').open() as stream:
  keys={r['Source_Key'] for r in csv.DictReader(stream)}
 assert {r[10] for k in COUNTS for r in rows(k)} <= keys
