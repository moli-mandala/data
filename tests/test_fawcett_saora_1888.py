"""Complete article proposal checks; no database build or canonical mutation."""
import csv,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/fawcett_saora_1888'
def read(name):return [json.loads(x)for x in(P/name).read_text().splitlines()if x]
def inventory():return list(csv.reader((P/'proposal.csv').open())),read('proposal-audit.jsonl')
def test_full_article_and_control_scope():
 rows,audit=inventory();pages=read('article-page-census-first-reading-20260927.jsonl')
 assert {p['printed_page']for p in pages}==set(range(206,273)) and len(pages)==67
 assert len(audit)==164 and len({a['source_unit_key']for a in audit})==164
 assert len(rows)==128 and sum(bool(a['entry_keys'])for a in audit)==121
 assert sum(not a['entry_keys']for a in audit)==43
 assert all(not a['entry_keys']for a in audit if a['scope']!='Saora')
 assert all(a['entry_keys']for a in audit if a['scope']=='Saora')
def test_kinship_is_whole_and_ambiguities_not_guessed():
 rows,audit=inventory();a={u['source_unit_key']:u for u in audit};by={r[10]:r for r in rows}
 kin=[u for u in audit if u['section']=='kinship'];assert len(kin)==41
 assert all(all(by[k][2].startswith('Mian ' if u['printed_page']==226 else 'Main ')for k in u['entry_keys'])for u in kin)
 for tail,expected in [('03','Main kakona awobyan onnu thon,am,ba'),('06','Main obayana balarn vulengo'),('08','Main kakeyan ayyi navonu')]:
  assert by['fawcett1888saora:p227:kinship:'+tail][2]==expected
 u=a['fawcett1888saora:p226:kinship:32']
 assert [by[k][2]for k in u['entry_keys']]==['Mian akeyan oba onnu','Mian woba wanno']
 assert all(by[k][3]=='my brother’s son’s' and 'uncertain'in by[k][14].split()for k in u['entry_keys'])
 for key in ['p226:kinship:08','p226:kinship:09','p227:kinship:07']:
  row=by['fawcett1888saora:'+key];assert '[source ditto]'in row[3] and 'uncertain'in row[14].split()
 assert not any(' or 'in r[2]for r in rows)
def test_literal_distinctions_and_source_qualifications():
 rows,_=inventory();by={r[10]:r for r in rows};prefix='fawcett1888saora:'
 assert by[prefix+'p242:numeral:09'][2]=='Thīmji'
 assert by[prefix+'p242:numeral:12'][2]=='Mïgŭl'
 assert by[prefix+'p242:numeral:13'][2]=='Mïgŭl Bōī'
 assert by[prefix+'p250:prose:first-feast'][2]=='Limmă'
 assert by[prefix+'p208:prose:arrow'][2]=='Sherràka'
 assert by[prefix+'p260:prose:earth'][2]=='Lŏbō'
 assert 'uncertain'in by[prefix+'p247:prose:soul'][14].split()
 assert all(not r[4]and not r[5]for r in rows)
def test_profile_tags_and_source_graph_are_complete():
 from segments import Tokenizer
 from tags import GRAMMATICAL_TAGS,GENDER_TAGS
 rows,audit=inventory();by={r[10]:r for r in rows};convert=Tokenizer(str(P/'proposal-profile.txt'))
 assert len(by)==len(rows)
 for r in rows:
  assert len(r)==15 and r[0]=='so' and not r[8]and not r[9]
  assert '�'not in convert(r[2],column='IPA')
  import unicodedata
  assert convert(unicodedata.normalize('NFC',r[2]),column='IPA') == convert(unicodedata.normalize('NFD',r[2]),column='IPA')
  assert set(r[14].split())<=GRAMMATICAL_TAGS|GENDER_TAGS
  assert 'n'not in r[14].split() # no accidental neuter tag from noun shorthand
  assert r[7].startswith('fawcett1888saora[p. ')and ';'not in r[7]
  if r[11]:assert r[11]in by and r[11]!=r[10]and by[r[11]][3]==r[3]
 assert sum(bool(r[11])for r in rows)==7
 assert {k for u in audit for k in u['entry_keys']}==set(by)

def test_actual_parser_obeys_declared_route_without_splitting_source_commas(monkeypatch, tmp_path):
 import io
 import make_cldf,source_meta
 from segments import Tokenizer
 meta=source_meta.SourceMeta([P/'20260927-fawcett-saora.yaml'])
 monkeypatch.setattr(source_meta,'load',lambda:meta)
 convert=Tokenizer(str(P/'proposal-profile.txt'))
 monkeypatch.setitem(make_cldf.convertors,'fawcett-saora-1888',convert)
 errors=io.StringIO()
 staged=tmp_path/'other/forms/20260927-fawcett-saora.csv'
 staged.parent.mkdir(parents=True)
 staged.write_bytes((P/'proposal.csv').read_bytes())
 parsed,stats=make_cldf.parse_file(str(staged),errors)
 raw,_=inventory();by={r[10]:r for r in raw}
 assert not errors.getvalue()
 assert len(parsed)==len(raw)==128
 assert {r.entry_key for r in parsed}==set(by)
 for r in parsed:
  source=by[r.entry_key]
  assert r.old_form==source[2]
  assert r.form==convert(source[2],column='IPA').replace(' ','').replace('#',' ')
  assert r.source==source[7] and r.lang=='so'
 assert stats['converted']==128
 assert meta.flag('fawcett1888saora','forms','split_alternates',True) is False


def test_installer_requires_exact_final_approval(monkeypatch):
 import runpy
 import pytest
 module=runpy.run_path(str(P/'import_source.py'),run_name='fawcett_import_test')
 validate=module['validate']
 rows,audit,_=validate()
 assert len(rows)==128 and len(audit)==164
 monkeypatch.setitem(validate.__globals__,'APPROVAL_NAME','never-approved-test-fixture.json')
 with pytest.raises(ValueError,match='approval has not been recorded'):
  validate(require_approval=True)
