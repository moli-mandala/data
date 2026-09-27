"""Focused full source staging checks without DB builds."""
import csv,importlib.util,io,json,unicodedata
from pathlib import Path
from collections import Counter
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_outer_siraji_1908'
s=importlib.util.spec_from_file_location('outer_full',P/'import_source_full.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def staged():return list(csv.reader((P/'literal-staged.csv').open()))
def row(p,sec,i,answer=1):
 key=f'bailey1908outersiraji:p{p}:{sec}:item:{i}'+(f':answer{answer}' if answer>1 else '')
 return {r[10]:r for r in staged()}[key]
def test_full_scope_and_reproducibility():
 r,a=m.generate()
 assert r==staged()
 assert a==[json.loads(x) for x in (P/'full-staged-audit.jsonl').read_text().splitlines()]
 assert len(r)==450 and len(a)==367
 assert len({x[10] for x in r})==450 and len({x['source_cell_key'] for x in a})==367
 assert {x['printed_page'] for x in a}==set(range(36,44))
 assert Counter(x['status'] for x in a)=={'ingested':364,'exclude_pattern':3}
 assert sum(x['section']=='lexical' for x in a)==123
 assert sum(x['section']=='cardinal' for x in a)==46
 assert sum(x['section']=='ordinal' for x in a)==9
 assert sum(x['section']=='specimen' for x in a)==5

def test_old_keys_and_suffix_expansions():
 old=list(csv.reader((P/'legacy-pilot.csv').open()))
 assert len(old)==100 and {r[10] for r in old}<={r[10] for r in staged()}
 assert row(41,'left',23)[2]=='bākrī' and row(41,'left',26)[2]=='kūkrī'
 assert row(41,'right',3)[2]=='braiḷī'
 assert row(41,'left',8)[2]=='s̲h̲ōrī' and row(41,'left',14)[2]=='shōrī'
 assert 'Source pattern:' in row(41,'left',23)[6]

def test_literal_signs_and_italic_source_notes():
 assert row(42,'left',19)[2]=='dzuth'
 assert row(42,'left',25)[2]=='bēdzau'
 assert row(42,'right',29)[2]=='nī̃ṇu'
 assert row(43,'cardinal:left',24)[2]=='saĩtī'
 assert 'italicizes u' in row(41,'right',10)[6]
 assert 'italicizes u' in row(38,'pronoun-1sg',5)[6]

def test_paradigms_labels_and_unprinted_forms():
 assert row(37,'noun-horse',2)[2]=='ghōṛĕau' and row(37,'noun-horse',2,2)[2]=='ghōṛĕē'
 assert 'f' in row(37,'noun-horse',2,2)[14].split()
 assert row(38,'pronoun-3sg',2,2)[2]=='tĕssō' and 'f' in row(38,'pronoun-3sg',2,2)[14].split()
 assert sum(':noun-sheep:' in r[10] for r in staged())==3
 assert 'Plural column contains ellipses' in row(37,'noun-sheep',1)[6]
 assert 'Subject in agent case' in row(40,'beat-past',1)[6]
 assert {'pret','ipfv','1sg'}<=set(row(39,'fall-imperfect','1sg')[14].split())

def test_whole_translated_expressions():
 assert row(40,'ability-expression',1)[2]=='mērē nĕhī̃ ḍēundō'
 assert row(40,'ability-expression',1,2)[2]=='mērē bhŏlē nĕhī̃ ḍēundō'
 assert row(41,'necessity-expression',1)[2]=='mū̃ kāllā ḍēuṇu'
 assert row(43,'specimen',20)[2]=='kaurō s̲h̲ōrū tā pitshu hāṇḍdō lagō aundō?'
 assert all('sentential' in row(43,'specimen',i)[14] for i in [6,7,17,19,20])

def test_literal_profile_and_scoped_parser():
 import tags,make_cldf
 r=staged();t=Tokenizer(str(P/'literal-profile-staged.txt'))
 assert set(' '.join(x[14] for x in r).split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
 for x in r:
  assert len(x)==15 and x[0]=='OuterSiraji'
  actual=unicodedata.normalize('NFC',t(x[2],column='IPA').replace(' ','').replace('#',' '))
  assert actual==x[2].replace('w','v').replace('ṅ','ŋ').replace('.','').replace('?','')
 old=make_cldf.convertors.get('bailey-outer-siraji-1908')
 try:
  make_cldf.convertors['bailey-outer-siraji-1908']=t
  e=io.StringIO();out,stats=make_cldf.parse_file(str(P/'literal-staged.csv'),e,name='20260925-bailey-outer-siraji')
  assert not e.getvalue() and len(out)==stats['converted']==450
 finally:
  if old is None:make_cldf.convertors.pop('bailey-outer-siraji-1908',None)
  else:make_cldf.convertors['bailey-outer-siraji-1908']=old
