"""Whole Bhadrawahi chapter source-stage checks, without a database build."""
import csv,importlib.util,io,json,unicodedata
from pathlib import Path
from collections import Counter
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_bhadrawahi_1908'
s=importlib.util.spec_from_file_location('bhadrawahi_full',P/'import_source_full.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def staged():return list(csv.reader((P/'full-staged.csv').open()))
def test_scope_identity_and_determinism():
 r,a=m.generate();assert r==staged()
 assert a==[json.loads(x) for x in (P/'full-staged-audit.jsonl').read_text().splitlines()]
 assert len(a)==623 and len(r)==709 and len({x[10] for x in r})==709
 assert {x['printed_page'] for x in a}==set(range(57,68))|{53,54,55}
 assert Counter(x['status'] for x in a)=={'ingested':615,'exclude_pattern':3,'exclude_other_language':5}
 counts=Counter(x['section'] for x in a)
 assert [counts[x] for x in ('glossary','cardinal','ordinal','fraction','sentences')]==[149,27,8,6,22]
 old=list(csv.reader((P/'legacy-pilot.csv').open()))
 assert len(old)==23 and {x[10] for x in old}<={x[10] for x in r}
 assert all(x['scan_page']==x['printed_page']+114 for x in a)
def test_source_witnesses_and_grammar():
 r={x[10]:x for x in staged()};prefix='bailey1908bhadrawahi:'
 assert r[prefix+'p59:pronoun-correlative-sg:item:3'][2]=='tus'
 assert r[prefix+'p59:pronoun-correlative-sg:item:3:answer:2'][2]=='tas̲h̲ jaū̃'
 assert 'possibly correlative' in r[prefix+'p59:pronoun-correlative-sg:item:1'][6]
 assert r[prefix+'p57:noun-name-introduction:item:1'][2]=='naū'
 assert r[prefix+'p57:noun-name:item:1'][2]=='naũ'
 assert r[prefix+'p54:introduction-correspondences:item:1'][2]=='ḍhḷubbū'
 assert r[prefix+'p64:1:item:27'][2]=='ḍhḷabbu'
 assert len([x for x in r.values() if not x[3]])==2
 assert all('uncertain' in x[14] and 'separate lexical gloss' in x[6] for x in r.values() if not x[3])
 assert all('multiword-expression' in x[14] for x in r.values() if ':sentences:' in x[10])
 assert r[prefix+'p66:cardinal:item:4'][2]=='tse͞uūr'
 assert r[prefix+'p65:left:item:40'][2]=='ṭhaṇḍū'
 assert r[prefix+'p64:epenthesis-examples:item:4'][2]=='bitshaṛulō'
 assert r[prefix+'p64:epenthesis-examples:item:4:answer:2'][2]=='bitshuṛailai'
 assert r[prefix+'p67:sentences:item:9'][2].endswith('kutṭū')
 assert 'uncertain' in r[prefix+'p60:adverb-prose:item:6'][14]
 assert 'uncertain' not in r[prefix+'p60:adverb-prose:item:6:answer:2'][14]
 assert r[prefix+'p60:adverb-prose:item:6'][2]=='un sārē'
 assert len([x for x in r.values() if x[3]=='being'])==5
def test_profile_tags_and_scoped_parse():
 import tags,make_cldf
 t=Tokenizer(str(P/'literal-profile-staged.txt'));rows=staged()
 for r in rows:
  assert len(r)==15 and r[0]=='bhad'
  assert set(r[14].split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
  actual=unicodedata.normalize('NFC',t(r[2],column='IPA').replace(' ','').replace('#',' '))
  assert actual==r[2].replace('w','v').replace('ṅ','ŋ')
 old=make_cldf.convertors.get('bailey-bhadrawahi-1908')
 try:
  make_cldf.convertors['bailey-bhadrawahi-1908']=t
  e=io.StringIO();parsed,stats=make_cldf.parse_file(str(P/'full-staged.csv'),e,name='20260925-bailey-bhadrawahi')
  assert not e.getvalue() and len(parsed)==stats['converted']==709
 finally:
  if old is None:make_cldf.convertors.pop('bailey-bhadrawahi-1908',None)
  else:make_cldf.convertors['bailey-bhadrawahi-1908']=old
