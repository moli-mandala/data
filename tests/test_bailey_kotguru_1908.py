"""Focused whole-chapter source regressions; no database build."""
import csv,importlib.util,io,json,unicodedata
from pathlib import Path
from collections import Counter
from segments.tokenizer import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_kotguru_1908'
s=importlib.util.spec_from_file_location('kotguru_full',P/'import_source.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def staged():return list(csv.reader((DATA/'data/other/forms/20260925-bailey-kotguru.csv').open()))
def row(p,sec,i,answer=1):
 key=f'bailey1908kotguru:p{p}:{sec}:item:{i}'+(f':answer{answer}' if answer>1 else '')
 return {r[10]:r for r in staged()}[key]
def test_full_scope_and_reproducibility():
 r,a=m.generate()
 assert r==staged()
 assert a==[json.loads(x) for x in (P/'audit.jsonl').read_text().splitlines()]
 assert len(r)==610 and len(a)==487
 assert len({x[10] for x in r})==610 and len({x['source_cell_key'] for x in a})==487
 assert {x['printed_page'] for x in a}==set(range(25,34))
 assert sum(x['section']=='lexical' for x in a)==151
 assert sum(x['section']=='cardinal' for x in a)==29
 assert sum(x['section']=='ordinal' for x in a)==17
 assert sum(x['section']=='numeral_note' for x in a)==4
 assert sum(x['section']=='specimen' for x in a)==22

def test_pilot_keys_and_full_suffix_expansions():
 old=list(csv.reader((P/'legacy-pilot.csv').open()))
 assert len(old)==112 and {r[10] for r in old}<={r[10] for r in staged()}
 assert row(30,'right',2)[2]=='bākrī'
 assert row(30,'right',11)[2]=='murgī' and row(30,'right',11,2)[2]=='kukkhṛī'
 assert row(30,'left',7)[2]=='chōṭī' and row(30,'left',7,2)[2]=='tshōṭī'
 assert 'Source pattern:' in row(30,'right',11)[6]

def test_typographic_distinctions():
 assert row(31,'right',6)[2]=='bēdẓau'
 assert row(31,'left',40)[2]=='daihṛō'
 assert row(31,'left',22)[2]=='gīhū̃'
 assert row(29,'do-head',1)[2]=='kŏrnuu' and row(29,'bring-past',1)[2]=='āṇuu'
 assert row(29,'go-ja-past',1,2)[2]=='gēĭ'

def test_grammar_and_source_labels():
 assert row(25,'noun-horse',7)[2]=='gōhṛĕā'
 assert 'voc' in row(25,'noun-horse',7)[14].split()
 assert row(29,'beat-past',1)[2]=='mārau'
 assert 'erg' not in row(29,'beat-past',1)[14].split()
 assert 'Subject in agent case' in row(29,'beat-past',1)[6]
 assert 'caus' in row(32,'right',9)[14].split()

def test_whole_sentences_and_printed_alternatives():
 assert row(33,'specimen',2)[2]=='ēū gōhṛĕai kai umar ā?'
 assert row(33,'specimen',2,2)[2]=='ēū gōhṛĕai kai umar āsā?'
 assert row(33,'specimen',19)[2]=='mūkā āgdī hāṇḍau.'
 assert row(33,'specimen',19,2)[2]=='mūkā āgdē hāṇḍau.'
 assert all('sentential' in row(33,'specimen',i)[14] for i in range(1,23))

def test_literal_profile_and_scoped_parser():
 import tags,make_cldf
 r=staged();t=Tokenizer(str(DATA/'conversion/bailey-kotguru-1908.txt'))
 assert set(' '.join(x[14] for x in r).split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS
 for x in r:
  assert len(x)==15 and x[0]=='Kotguru'
  actual=unicodedata.normalize('NFC',t(x[2],column='IPA').replace(' ','').replace('#',' '))
  assert actual==x[2].replace('w','v').replace('ṅ','ŋ').replace('.','').replace('?','')
 old=make_cldf.convertors.get('bailey-kotguru-1908')
 try:
  make_cldf.convertors['bailey-kotguru-1908']=t
  e=io.StringIO();out,stats=make_cldf.parse_file(str(DATA/'data/other/forms/20260925-bailey-kotguru.csv'),e,name='20260925-bailey-kotguru')
  assert not e.getvalue() and len(out)==stats['converted']==610
 finally:
  if old is None:make_cldf.convertors.pop('bailey-kotguru-1908',None)
  else:make_cldf.convertors['bailey-kotguru-1908']=old

def test_independent_audit_and_specific_hold():
 import hashlib,source_meta,profile_policy
 report=json.loads((P/'independent-audit-20260926-pass1.json').read_text())
 assert report['material_literal_or_gloss_errors']==0 and report['sample_size']==20
 add=json.loads((P/'independent-audit-20260926-pass1-addendum.json').read_text())
 assert add['sampled_units_unchanged']==20
 assert add['post_adjudication_csv_sha256']==hashlib.sha256((DATA/'data/other/forms/20260925-bailey-kotguru.csv').read_bytes()).hexdigest()
 r,a=m.generate();holds=[x for x in a if x['status']=='hold_typography']
 assert len(holds)==1 and holds[0]['source_cell_key']=='bailey1908kotguru:p28:preposition-right:item:1'
 assert '650dpi' in holds[0]['reason'] and not holds[0]['entry_keys']
 assert Counter(x['status'] for x in a)=={'ingested':485,'exclude_pattern':1,'hold_typography':1}
 assert all('Pres. Cond. or Indic.' in x[6] for x in r if ':come-pres-conditional:' in x[10] or ':go-de-present:' in x[10])
 assert source_meta.SourceMeta().transcription('bailey1908kotguru',DATA/'data/other/forms/20260925-bailey-kotguru.csv','Kotguru')[0]=='bailey-kotguru-1908'
 assert 'bailey-kotguru-1908' not in profile_policy.audit({})
