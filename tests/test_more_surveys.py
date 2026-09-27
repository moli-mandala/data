import csv,importlib.util,json,re,unicodedata
from pathlib import Path
import pytest
from segments import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('more_surveys',ROOT/'data/other/forms/raw_data/more_surveys.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
COUNTS={'jharkhand':4407,'himachal':6442,'rajasthan':4567,'west-bengal':1835,'kisan':1067}
def rows(k):return list(csv.reader((ROOT/f'data/other/forms/20260911-more-{k}.csv').open()))
def profile(k,lang):return 'more-ipa' if k=='kisan' or k=='himachal' and lang in {'sirm','pan','dog'} else 'more-ascii'
@pytest.mark.parametrize('k',COUNTS)
def test_installed_counts_registry_profile(k):
 from tags import GRAMMATICAL_TAGS,GENDER_TAGS
 langs={x['ID'] for x in csv.DictReader((ROOT/'cldf/languages.csv').open())};ds={x['Tag']:x['Language_ID'] for x in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
 rr=rows(k);assert len(rr)==COUNTS[k];assert len({x[10] for x in rr})==len(rr)
 toks={p:Tokenizer(str(ROOT/'conversion'/(p+'.txt'))) for p in ['more-ascii','more-ipa']}
 for r in rr:
  assert len(r)==15 and r[0] in langs and r[2] and r[3] and not r[1]
  assert r[2]==unicodedata.normalize('NFC',r[2]);assert '(cid:' not in r[2]
  assert '�' not in toks[profile(k,r[0])](r[2],column='IPA'),r[10]
  assert r[7].startswith(m.SOURCES[k]+'[p. ')
  for t in r[14].split():
   if t.startswith('dialect:'):assert ds[t]==r[0]
   else:assert t in GRAMMATICAL_TAGS|GENDER_TAGS,t
@pytest.mark.parametrize('k',COUNTS)
def test_all_raw_cells_accounted(k):
 a=[json.loads(l) for l in (m.RAW/(k+'-audit.jsonl')).read_text().splitlines()];assert len(a)==m.EXPECTED[k]
 emitted=[x['row'] for r in a for x in r['readings']];assert emitted==rows(k)
 assert len({x['entry_key'] for x in a})==len(a)
def test_missing_and_misattributed_sources():
 a=[json.loads(l) for l in (m.RAW/'darai-excluded-audit.jsonl').read_text().splitlines()]
 assert len(a)==1050 and all(x['same_ignoring_layout'] for x in a)
 missing=m.load('unavailable-sources.json');assert all(x['government_copy_byte_identical'] for x in missing.values())
 assert not (ROOT/'data/other/forms/20260911-more-darai.csv').exists()
def test_repeated_item_numbers_and_source_damage():
 a=m.load('himachal-cells.json');assert len([x for x in a if x['column']==0 and x['item']==327])==2
 keys=[m.cellkey('himachal',x) for x in a];assert len(keys)==len(set(keys))
 assert all(not m.parse('himachal',x) for x in a if x['images'])
 assert len([x for x in a if x['images']])==2
 assert not any(x['item']==341 for x in a)
def test_annotation_scope_and_line_wraps():
 def find(k,item,col):return [r for r in rows(k) if f':i{item}:c{col}:' in r[10]]
 assert [(r[2],r[3]) for r in find('jharkhand',38,7)]==[('engDa','my daughter'),('ningDa','your daughter'),('tangDa','his/her daughter')]
 assert [r[2] for r in find('kisan',4,3)]==['muɦərən','tseɦəra']
 assert find('kisan',8,3)[0][2]=='tʰ̪tʰ̪na'
 rr=find('rajasthan',24,1);assert 'sg' in rr[0][14].split() and 'pl' in rr[1][14].split()
 rr=find('jharkhand',31,1);assert 'Sikaripara' not in rr[0][14] and 'Sikaripara' in rr[2][14]
 assert [r[2] for r in find('rajasthan',104,0)]==['pAg','pAglya']
def test_profile_interpretations():
 def conv(p,s):return Tokenizer(str(ROOT/'conversion'/(p+'.txt')))(s,column='IPA').replace(' ','').replace('#',' ')
 assert conv('more-ascii','pAhaD')=='pahāḍ'  # the volume's key: A is the short vowel, plain a the long one
 assert conv('more-ipa','ɖaːɡ')=='ḍāg'
 assert conv('more-ascii','bãdh(a)')=='bā̃dʰ(ā)'
 assert conv('more-ascii','۠')=='۠' # ambiguous printed mark retained, not interpreted
 assert conv('more-ipa','ɟ')=='j' # the IPA palatal stop is the house j

def test_reproducible(tmp_path):
 m.build(tmp_path)
 for k in COUNTS:assert (tmp_path/f'20260911-more-{k}.csv').read_bytes()==(ROOT/f'data/other/forms/20260911-more-{k}.csv').read_bytes()

def test_compiled_preserves_every_key_and_citation():
 keys={r[10]:r for k in COUNTS for r in rows(k)}
 links=list(csv.DictReader((ROOT/'cldf/form-source-keys.csv').open()));aliases={r['Legacy_ID']:r['Form_ID'] for r in csv.DictReader((ROOT/'cldf/form-id-aliases.csv').open())}
 ids={r['Source_Key']:aliases[r['Legacy_ID']] for r in links if r['Source_Key'] in keys};assert ids.keys()==keys.keys()
 wanted=set(ids.values())
 forms={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open()) if r['ID'] in wanted}
 for key,ident in ids.items():
  row=keys[key];f=forms[ident];assert row[7] in f['Source'];assert f['Original'] and f['Form'] and '�' not in f['Form']
  assert set(row[14].split())<=set(f['Tags'].split())
 refs={r['ID'] for r in csv.DictReader((ROOT/'cldf/references.csv').open())};assert set(m.SOURCES.values())<=refs
