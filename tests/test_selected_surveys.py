import csv,importlib.util,json,unicodedata
from pathlib import Path
import pytest
from segments import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('selected_surveys',ROOT/'data/other/forms/raw_data/selected_surveys.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def rows(k):return list(csv.reader((ROOT/f'data/other/forms/20260911-selected-{k}.csv').open()))
@pytest.mark.parametrize('k',m.SOURCES)
def test_installed_registry_profile_and_schema(k):
 from tags import GRAMMATICAL_TAGS,GENDER_TAGS
 langs={x['ID'] for x in csv.DictReader((ROOT/'cldf/languages.csv').open())};ds={x['Tag']:x['Language_ID'] for x in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
 rr=rows(k);report=m.load('report.json')[k]
 assert len(rr)==report['installed_rows'];assert len({r[10] for r in rr})==len(rr)
 tok=Tokenizer(str(ROOT/'conversion'/f'selected-{k}.txt'))
 for r in rr:
  assert len(r)==15 and r[0] in langs and r[2] and r[3] and not r[1]
  assert not any(r[i] for i in [4,5,8,11,12,13])
  assert r[2]==unicodedata.normalize('NFC',r[2]);assert '(cid:' not in r[2]
  assert '�' not in tok(r[2],column='IPA'),(r[10],r[2])
  assert r[7].startswith(m.SOURCES[k]+'[p. ')
  assert 'uncertain' in r[14].split()
  for t in r[14].split():
   if t.startswith('dialect:'):assert ds[t]==r[0]
   else:assert t in GRAMMATICAL_TAGS|GENDER_TAGS,t
@pytest.mark.parametrize('k',m.SOURCES)
def test_complete_audit(k):
 a=[json.loads(l) for l in (m.RAW/(k+'-audit.jsonl')).read_text().splitlines()]
 assert len(a)==m.EXPECTED[k];assert len({x['entry_key'] for x in a})==len(a)
 assert [x['row'] for r in a for x in r['readings']]==rows(k)
 assert all(x['review'] for r in a for x in r['readings'])
def test_pinned_counts_and_missing_glyphs():
 assert [len(rows(k)) for k in ['angika','majhi','koraga','orissa']]==[1076,1055,1369,5713]
 assert sum(not m.parse('angika',r) for r in m.load('angika-cells.json'))==2
 assert sum(not m.parse('majhi',r) for r in m.load('majhi-cells.json'))==1
def test_koraga_glyphs_and_undefined_lect():
 import sys
 sys.path.insert(0,str(m.RAW));import koraga
 rr={(r['printed_page'],r['item']):koraga.parse(r) for r in koraga.records()}
 assert rr[95,32][0]['form']=='ki:rɨ' and rr[95,32][0]['lect']==''
 assert any(r['form']=='ta:lɨ' and not r['lect'] for r in rr[101,19])
 assert {r['form'] for r in rr[109,27]}=={'magalɨ','magaḷɨ'}
 assert rr[116,21][0]['form']=='hiṇḍi'
 assert len(m.load('koraga-glyph-review.json'))==857
 assert not m.load('koraga-glyph-decisions.json')['needs_larger_crop']
 assert all(not r[1] and not any(r[11:14]) for r in rows('koraga'))
 assert sum('dialect:' not in r[14] for r in rows('koraga'))==4
def test_orissa_category_boundaries_and_controls():
 import sys
 sys.path.insert(0,str(m.RAW));import orissa
 rr,excluded=orissa.records();assert len(rr)==5065 and excluded
 assert {(r['item'],r['column']) for r in rr}=={(i,c) for i in range(1,1014) for c in range(5)}
 assert next(r for r in rr if r['item']==665)['pdf_page']==243
 assert next(r for r in rr if r['item']==666)['pdf_page']==244
 assert next(r for r in rr if r['item']==543)['pdf_page']==239
 assert not {'Gondi','Kui'}&{r[0] for k in m.SOURCES for r in rows(k)}
 assert sum(not orissa.parse(r) for r in rr)==22
def test_orissa_reviewed_error_classes():
 import sys
 sys.path.insert(0,str(m.RAW));import orissa
 rr={(r['item'],r['column']):r for r in m.load('orissa-cells.json')}
 def ps(i,c):return orissa.parse(rr[i,c])
 assert [p['form'] for p in ps(696,3)]==['kObaT kulbar','ãki kulbar','munapiTeibar']
 assert [p['form'] for p in ps(62,3)]==['pila','nuna']
 assert ps(698,3)[0]['form']=='gẽJbar' and ps(765,3)[0]['form']=='ũCli pODbar'
 assert ps(845,0)[0]['form']=='hOsiba'
 assert len(ps(173,0))==2 and all('poisionous' not in p['form'] for p in ps(173,0))
 assert ps(139,0)[0]['tags']==['m'] and ps(140,0)[0]['tags']==['f']
 assert [p['tags'] for p in ps(1006,3)]==[['sg'],['pl']]
 assert ps(554,1)[0]['form']=='baêgOni'
def test_consequential_profile_mappings():
 tok=Tokenizer(str(ROOT/'conversion/selected-koraga.txt'))
 assert tok('ki:rɨ',column='IPA').replace(' ','')=='kīrɨ'
 tok=Tokenizer(str(ROOT/'conversion/selected-orissa.txt'))
 assert tok('TODcj',column='IPA').replace(' ','')=='ṭɔḍčǰ'
 assert tok('gẽJbar',column='IPA').replace(' ','')=='gẽJbar'
def test_reproducible(tmp_path):
 m.build(tmp_path)
 for k in m.SOURCES:assert (tmp_path/f'20260911-selected-{k}.csv').read_bytes()==(ROOT/f'data/other/forms/20260911-selected-{k}.csv').read_bytes()
def test_compiled_survival():
 keys={r[10]:r for k in m.SOURCES for r in rows(k)}
 aliases={r['Legacy_ID']:r['Form_ID'] for r in csv.DictReader((ROOT/'cldf/form-id-aliases.csv').open())}
 ids={r['Source_Key']:aliases[r['Legacy_ID']] for r in csv.DictReader((ROOT/'cldf/form-source-keys.csv').open()) if r['Source_Key'] in keys}
 assert ids.keys()==keys.keys()
 wanted=set(ids.values());forms={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open()) if r['ID'] in wanted}
 for key,ident in ids.items():
  r=keys[key];f=forms[ident];assert r[7] in f['Source'];assert f['Original']==r[2]
  assert set(r[14].split())<=set(f['Tags'].split());assert '�' not in f['Form']
 refs={r['ID'] for r in csv.DictReader((ROOT/'cldf/references.csv').open())};assert set(m.SOURCES.values())<=refs
