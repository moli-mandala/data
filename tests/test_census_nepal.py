import csv,importlib.util,json,re,unicodedata
from pathlib import Path
import pytest
from segments import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('census_nepal',ROOT/'data/other/forms/raw_data/census_nepal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
COUNTS={'tamil-nadu':5995,'uttar-pradesh':5931,'bihar':4177,'sikkim2':1729,'danuwar':1051,'tharu':1520}
def rows(k):return list(csv.reader((ROOT/f'data/other/forms/20260911-census-{k}.csv').open()))
@pytest.mark.parametrize('k',COUNTS)
def test_counts_provenance_registry_and_symbols(k):
 rr=rows(k);assert len(rr)==COUNTS[k];assert len({r[10] for r in rr})==len(rr)
 langs={r['ID'] for r in csv.DictReader((ROOT/'cldf/languages.csv').open())};ds={r['Tag']:r['Language_ID'] for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
 tok=Tokenizer(str(ROOT/'conversion'/(m.PROFILES[k]+'.txt')))
 for r in rr:
  assert len(r)==15 and r[0] in langs and r[2] and r[3] and not r[1]
  assert r[7].startswith(m.SOURCES[k]+'[')
  assert not any(b in r[2] for b in ['�','(cid:','#VALUE!','to be replaced'])
  assert not re.fullmatch(r'[()\s/]+',r[2])
  assert r[2]==unicodedata.normalize('NFC',r[2])
  assert '�' not in tok(r[2],column='IPA'),r[10]
  for t in r[14].split():
   if t.startswith('dialect:'):assert ds[t]==r[0]
@pytest.mark.parametrize('k',COUNTS)
def test_every_raw_cell_accounted(k):
 aa=[json.loads(l) for l in (m.RAW/(k+'-audit.jsonl')).read_text().splitlines()]
 assert len(aa)==m.EXPECTED[k]+(12 if k in ('tamil-nadu','uttar-pradesh') else 0)
 installed=[x['entry_key'] for a in aa for x in a.get('readings',[]) if x['status']=='unlinked']
 assert set(installed)=={r[10] for r in rows(k)}
 assert len(installed)==COUNTS[k]
def test_image_patches_cover_every_embedded_cell():
 pp=m.load('image-transcriptions.json')
 for k,n in [('tamil-nadu',24),('sikkim2',124)]:
  cells={str(x['item'])+':'+str(x['column']) for x in m.load(k+'-cells.json') if x['images']}
  assert len(cells)==n and cells==pp[k].keys()
 rr={r[10]:r for r in rows('sikkim2')}
 assert rr['sikkim2:i1:c0:v1'][2]=='hawa'
 assert rr['sikkim2:i1:c1:v1'][2]=='hAwe'
 assert rr['sikkim2:i1:c1:v2'][2]=='hawa'
def test_omissions_variants_and_annotations():
 for k in ('tamil-nadu','uttar-pradesh'):assert not any(':i94:' in r[10] for r in rows(k))
 rr={r[10]:r for r in rows('sikkim2')}
 assert rr['sikkim2:i55:c0:v1'][2]=='moTo' and 'm' in rr['sikkim2:i55:c0:v1'][14].split()
 assert rr['sikkim2:i55:c0:v2'][2]=='moTi' and 'f' in rr['sikkim2:i55:c0:v2'][14].split()
 assert all('to float' not in r[2] for r in rows('tamil-nadu'))
 assert next(r for r in rows('danuwar') if r[10]=='danuwar:i87:c3:v1')[2]=='murgabəcca'
def test_reused_tharu_has_citations_without_duplicate_readings():
 reuse=m.load('tharu-reuse.json');assert sum(len(v) for v in reuse.values())==761
 old={r[10]:r for r in csv.reader((ROOT/'data/other/forms/20260813-kochila-tharu.csv').open())}
 for key,cc in reuse.items():assert all(c in old[key][7].split(';') for c in cc)
 assert 'kochila:241:kochila_morang_east:1' in reuse # workbook item 240 is numbered 241 in the publication.
def test_profile_interpretations():
 def conv(p,s):return Tokenizer(str(ROOT/'conversion'/(p+'.txt')))(s,column='IPA').replace(' ','').replace('#',' ')
 assert conv('census-ipa','ɖaːɡ')=='ḍāg'
 assert conv('census-ascii','pAhaD')=='pəhaḍ'
 assert conv('census-danuwar','t̺auko')=='t̺auko'
def test_reproducible(tmp_path):
 m.build(tmp_path)
 for k in COUNTS:assert (tmp_path/f'20260911-census-{k}.csv').read_bytes()==(ROOT/f'data/other/forms/20260911-census-{k}.csv').read_bytes()
def test_compiled_rows_and_graph():
 keys={r[10] for k in COUNTS for r in rows(k)}
 with (ROOT/'cldf/form-source-keys.csv').open() as f:links=list(csv.DictReader(f))
 found={r['Source_Key'] for r in links}
 with (ROOT/'cldf/form-id-aliases.csv').open() as f:aliases={r['Legacy_ID']:r['Form_ID'] for r in csv.DictReader(f)}
 new_ids={aliases[r['Legacy_ID']] for r in links if r['Source_Key'] in keys}
 assert keys<=found
 ids=set();count=0
 with (ROOT/'cldf/forms.csv').open() as f:
  for r in csv.DictReader(f):
   if any(s+'[' in r['Source'] for s in m.SOURCES.values()):
    assert r['Form'] and r['Original'] and '�' not in r['Form'];ids.add(r['ID'])
    if any(s+'[' in r['Source'] for s in list(m.SOURCES.values())[:-1]):assert r['Status']=='unlinked'
 # The census surveys assert no etymologies; edges on their forms may only come from the reviewed per-source sidecar (the 2026-09 joint SIL review linked Tharu cells).
 import sys;sys.path.insert(0,str(ROOT));from etymology_assignments import read_assignments
 reviewed={r['Form_ID'] for r in read_assignments()}
 with (ROOT/'cldf/edges.csv').open() as f:assert not any(r['Child_ID'] in new_ids and r['Child_ID'] not in reviewed for r in csv.DictReader(f))

def test_workbook_extraction():
 assert m.extract_workbook()==m.load("tharu-cells.json")
