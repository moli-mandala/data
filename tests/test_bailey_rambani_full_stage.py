"""Rambani full-chapter staging regressions, without database generation."""
import csv,importlib.util,io,json,unicodedata
from pathlib import Path
from collections import Counter
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_rambani_1908'
s=importlib.util.spec_from_file_location('rambani_full',P/'import_source_full.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def staged():return list(csv.reader((P/'full-staged.csv').open()))
def row(page,section,item,answer=1):
 k=f'bailey1908rambani:p{page}:{section}:item:{item}'+(f':answer:{answer}' if answer>1 else '')
 return {r[10]:r for r in staged()}[k]
def test_full_scope_and_determinism():
 r,a=m.generate();assert r==staged()
 assert a==[json.loads(x) for x in (P/'full-staged-audit.jsonl').read_text().splitlines()]
 assert len(a)==214 and len(r)==249 and len({x[10] for x in r})==249
 assert {x['printed_page'] for x in a}==set(range(46,51))
 assert Counter(x['status'] for x in a)=={'ingested':212,'exclude_pattern':1,'exclude_other_language':1}
 assert [int(x['item']) for x in a if x['section']=='glossary']==list(range(1,101))
 assert [int(x['item']) for x in a if x['section']=='specimens']==list(range(1,23))
 assert all(x['scan_page']==x['printed_page']+224 for x in a)
def test_legacy_identity_and_literal_recovery():
 old=list(csv.reader((P/'legacy-pilot.csv').open()))
 assert len(old)==62 and {r[10] for r in old}<={r[10] for r in staged()}
 for p,c,i,f in [(48,'left',6,'s̲h̲ĕ'),(48,'left',8,'aṭh'),(48,'right',39,'kāmŭ'),(49,'left',83,'ghōṛă'),(49,'left',84,'ghōrī'),(49,'right',100,'harn')]:assert row(p,c,i)[2]==f
 assert '28' in row(48,'right',38)[6]
def test_grammar_and_whole_expressions():
 assert row(46,'noun-woman',1)[2]=='zanānã'
 assert row(47,'adjective-good',1)[2]=='caŋgō'
 assert row(47,'go-past',4)[2]=='gēŭsam'
 assert '1sg' not in row(47,'beat-perfect',1)[14].split()
 assert row(50,'specimens',19)[2]=='mī agar cal'
 assert 'multiword-expression' in row(50,'specimens',19)[14]
 assert 'Urdu' in row(47,'compound-example',1)[6]
def test_profile_and_scoped_parse():
 import tags,make_cldf
 r=staged();t=Tokenizer(str(P/'literal-profile-staged.txt'))
 for x in r:
  assert len(x)==15 and x[0]=='ram'
  assert set(x[14].split())<=(tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS)
  actual=unicodedata.normalize('NFC',t(x[2],column='IPA').replace(' ','').replace('#',' '))
  assert actual==x[2].replace('w','v').replace('ṅ','ŋ')
 old=make_cldf.convertors.get('bailey-rambani-1908')
 try:
  make_cldf.convertors['bailey-rambani-1908']=t
  e=io.StringIO();out,stats=make_cldf.parse_file(str(P/'full-staged.csv'),e,name='20260925-bailey-rambani')
  assert not e.getvalue() and len(out)==stats['converted']==249
 finally:
  if old is None:make_cldf.convertors.pop('bailey-rambani-1908',None)
  else:make_cldf.convertors['bailey-rambani-1908']=old

def test_audit_remediation_h_and_feminine_agreement():
 assert row(50,'specimens',22)[2]=='gāma saṇi kē̃tsī haṭiăbālă thā̃'
 for s,i in [('pronoun-singular-3',2),('pronoun-plural-1',2),('pronoun-plural-2',2),('pronoun-plural-3',2)]:
  assert 'f' in row(47,s,i,2)[14].split()
  assert 'f' not in row(47,s,i,1)[14].split()
