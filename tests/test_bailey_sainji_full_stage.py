"""Whole Sainji chapter staging checks, without a database build."""
import csv,importlib.util,io,json,unicodedata
from pathlib import Path
from collections import Counter
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_sainji_1908'
s=importlib.util.spec_from_file_location('sainji_full',P/'import_source_full.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def staged():return list(csv.reader((P/'full-staged.csv').open()))
def test_scope_identity_and_determinism():
 r,a=m.generate();assert r==staged()
 assert a==[json.loads(x) for x in (P/'full-staged-audit.jsonl').read_text().splitlines()]
 assert len(a)==241 and len(r)==270 and len({x[10] for x in r})==270
 assert {x['printed_page'] for x in a}==set(range(52,57))
 assert Counter(x['status'] for x in a)=={'ingested':237,'exclude_pattern':3,'exclude_other_lect':1}
 assert Counter(x['section'] for x in a)['glossary']==46
 assert [int(x['item']) for x in a if x['section']=='numerals']==list(range(1,21))
 assert [int(x['item']) for x in a if x['section']=='sentences']==list(range(1,23))
 old=list(csv.reader((P/'legacy-pilot.csv').open()))
 assert len(old)==55 and {x[10] for x in old}<={x[10] for x in r}
 assert all(x['scan_page']==x['printed_page']+22 for x in a)
def test_literal_recovery_and_grammar():
 r={x[10]:x for x in staged()}
 assert r['bailey1908sainji:glossary:55:1:5'][2]=='tshōrī'
 assert r['bailey1908sainji:glossary:55:1:7'][2]=='bauiḷd'
 assert r['bailey1908sainji:glossary:55:1:13'][2]=='païr'
 assert r['bailey1908sainji:glossary:55:2:2'][3]=='jungle'
 assert r['bailey1908sainji:glossary:55:2:2:answer2'][2]=='būṇ'
 assert 'italic' in r['bailey1908sainji:glossary:55:2:7'][6]
 assert r['bailey1908sainji:p52:noun-daughter:item:4'][2]=='bēṭīē'
 assert len([x for x in r.values() if x[3]==''])==4
 assert all('No English lexical gloss' in x[6] and 'uncertain' in x[14] for x in r.values() if not x[3])
 assert 'multiword-expression' in r['bailey1908sainji:p56:sentences:item:1'][14]
 assert r['bailey1908sainji:p52:pronoun-singular-3:item:2:answer:2'][3]=='she'
 assert 'be͞uhṇi' in r['bailey1908sainji:p56:sentences:item:6'][2]
 assert 'bauïhṇī' in r['bailey1908sainji:p56:sentences:item:12'][2]
 assert r['bailey1908sainji:numerals:56:2:19'][2]=='ṇīh'
 assert r['bailey1908sainji:numerals:56:2:20'][2]=='bīh'
def test_profile_tags_and_scoped_parse():
 import tags,make_cldf
 t=Tokenizer(str(P/'literal-profile-staged.txt'));rows=staged()
 for r in rows:
  assert len(r)==15 and r[0]=='sai'
  assert set(r[14].split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
  actual=unicodedata.normalize('NFC',t(r[2],column='IPA').replace(' ','').replace('#',' '))
  assert actual==r[2].replace('w','v').replace('ṅ','ŋ')
 old=make_cldf.convertors.get('bailey-sainji-1908')
 try:
  make_cldf.convertors['bailey-sainji-1908']=t
  e=io.StringIO();parsed,stats=make_cldf.parse_file(str(P/'full-staged.csv'),e,name='20260925-bailey-sainji')
  assert not e.getvalue() and len(parsed)==stats['converted']==270
 finally:
  if old is None:make_cldf.convertors.pop('bailey-sainji-1908',None)
  else:make_cldf.convertors['bailey-sainji-1908']=old
